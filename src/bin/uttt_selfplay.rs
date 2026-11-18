use std::collections::BTreeMap;
use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use az_nn::MlpNetwork;
use az_pipeline::uttt_adapter::{model_shape, EvaluatorKind, Mcts, RootNoise};
use az_pipeline::uttt_data::{SampleWriter, TrainingSample};
use az_pipeline::uttt_features::encode;
use az_pipeline::uttt_game::{Game, BOARD_CELLS};
use az_pipeline::uttt_schema::FEATURE_SIZE;

struct Sample {
    features: [u8; FEATURE_SIZE],
    policy: [f32; BOARD_CELLS],
    player: i8,
    value: f32,
}

struct CompletedGame {
    index: usize,
    samples: Vec<Sample>,
    winner: Option<i8>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let games = parse_arg(&args, "--games", 10);
    let simulations = parse_arg(&args, "--simulations", 200);
    let threads = parse_arg(&args, "--threads", default_threads()).max(1);
    let dirichlet_alpha = parse_float_arg(&args, "--dirichlet-alpha", 0.3);
    let dirichlet_epsilon = parse_float_arg(&args, "--dirichlet-epsilon", 0.25);
    let seed = parse_u64_arg(&args, "--seed", 0);
    let tree_reuse = !args.iter().any(|arg| arg == "--no-tree-reuse");
    let output = parse_path(&args, "--output", "data/uttt/selfplay.bin");
    let checkpoint = optional_path(&args, "--checkpoint");
    assert!(games > 0, "--games must be greater than zero");
    assert!(simulations > 0, "--simulations must be greater than zero");
    assert!(dirichlet_alpha > 0.0, "--dirichlet-alpha must be positive");
    assert!(
        (0.0..=1.0).contains(&dirichlet_epsilon),
        "--dirichlet-epsilon must be between zero and one"
    );
    let evaluator = if let Some(path) = checkpoint {
        EvaluatorKind::Checkpoint(Arc::new(
            MlpNetwork::load(path, model_shape()).expect("load self-play checkpoint"),
        ))
    } else {
        EvaluatorKind::Heuristic
    };

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let temporary_output = temporary_path(&output);

    println!(
        "self-play: games={games} simulations={simulations} threads={threads} tree_reuse={tree_reuse} dirichlet={dirichlet_alpha}/{dirichlet_epsilon} seed={seed} evaluator={} output={}",
        match &evaluator {
            EvaluatorKind::Heuristic => "heuristic",
            EvaluatorKind::Checkpoint(_) => "checkpoint",
        },
        output.display()
    );
    io::stdout().flush().expect("flush self-play header");

    let mut writer =
        SampleWriter::create(&temporary_output).expect("create temporary self-play output");
    let started = Instant::now();
    let progress_every = (games / 100).max(1);
    let mut total_positions = 0;
    let mut outcomes = [0; 3];

    let next_game = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let worker_count = threads.min(games.max(1));
    thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_game = Arc::clone(&next_game);
            let sender = sender.clone();
            let evaluator = evaluator.clone();
            scope.spawn(move || loop {
                let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                if game_index >= games {
                    break;
                }
                let (samples, winner) = generate_game(
                    simulations,
                    evaluator.clone(),
                    tree_reuse,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    game_seed(seed, game_index),
                );
                sender
                    .send(CompletedGame {
                        index: game_index,
                        samples,
                        winner,
                    })
                    .expect("send completed self-play game");
            });
        }
        drop(sender);

        let mut completed = 0;
        let mut next_to_write = 0;
        let mut pending = BTreeMap::new();
        for result in receiver {
            completed += 1;
            total_positions += result.samples.len();
            match result.winner {
                Some(0) => outcomes[0] += 1,
                Some(1) => outcomes[1] += 1,
                None => outcomes[2] += 1,
                _ => unreachable!(),
            }
            pending.insert(result.index, result.samples);
            while let Some(samples) = pending.remove(&next_to_write) {
                write_samples(&mut writer, samples);
                next_to_write += 1;
            }

            if completed == 1 || completed == games || completed % progress_every == 0 {
                let elapsed = started.elapsed();
                let eta = elapsed.mul_f64((games - completed) as f64 / completed as f64);
                println!(
                    "self-play: {completed}/{games} ({:.1}%) positions={} X/O/draw={}/{}/{} elapsed={} eta={}",
                    completed as f64 * 100.0 / games as f64,
                    total_positions,
                    outcomes[0],
                    outcomes[1],
                    outcomes[2],
                    format_duration(elapsed),
                    format_duration(eta)
                );
                io::stdout().flush().expect("flush self-play progress");
            }
        }
        assert_eq!(completed, games, "self-play workers stopped early");
        assert!(pending.is_empty(), "not all games were written");
    });
    writer.flush().expect("flush self-play output");
    drop(writer);
    std::fs::rename(&temporary_output, &output).expect("publish self-play output");
    println!("self-play: wrote {games} games to {}", output.display());
}

fn parse_arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_float_arg(args: &[String], name: &str, default: f32) -> f32 {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_u64_arg(args: &[String], name: &str, default: u64) -> u64 {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn game_seed(seed: u64, game_index: usize) -> u64 {
    let game = game_index as u64 + 1;
    if seed == 0 {
        game
    } else {
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ game
    }
}

fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, usize::from)
}

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn temporary_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .expect("output file name");
    output.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

fn optional_path<'a>(args: &'a [String], name: &str) -> Option<&'a Path> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(Path::new)
}

fn generate_game(
    simulations: usize,
    evaluator: EvaluatorKind,
    tree_reuse: bool,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    seed: u64,
) -> (Vec<Sample>, Option<i8>) {
    let mut game = Game::new();
    let mut samples = Vec::with_capacity(81);
    let mut search = Mcts::new(1.5, evaluator);
    let mut rng = Rng::new(seed);

    while !game.is_terminal() {
        let result = search.search_with_noise(
            &game,
            simulations,
            RootNoise {
                alpha: dirichlet_alpha,
                epsilon: dirichlet_epsilon,
                seed: (seed << 32) ^ rng.next_u32() as u64,
            },
        );
        let total = result.visits.iter().sum::<u32>().max(1) as f32;
        let mut policy = [0.0; BOARD_CELLS];
        for action in 0..BOARD_CELLS {
            policy[action] = result.visits[action] as f32 / total;
        }
        let encoded = encode(&game);
        let mut features = [0; FEATURE_SIZE];
        for (target, value) in features.iter_mut().zip(encoded) {
            *target = u8::from(value != 0.0);
        }
        samples.push(Sample {
            features,
            policy,
            player: game.current_player(),
            value: 0.0,
        });
        let action = if samples.len() <= 20 {
            sample_visits(&result.visits, &mut rng)
        } else {
            result.action
        };
        game.play(action);
        if tree_reuse {
            search.advance(action);
        } else {
            search.reset();
        }
    }

    let winner = game.winner();
    for sample in &mut samples {
        sample.value = game.outcome(sample.player);
    }
    (samples, winner)
}

fn write_samples(writer: &mut SampleWriter, samples: Vec<Sample>) {
    for sample in samples {
        writer
            .write(&TrainingSample {
                features: sample.features,
                policy: sample.policy,
                value: sample.value,
            })
            .expect("write training sample");
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

fn sample_visits(visits: &[u32; BOARD_CELLS], rng: &mut Rng) -> usize {
    let total = visits.iter().sum::<u32>();
    let mut choice = rng.next_u32() % total.max(1);
    for (action, &count) in visits.iter().enumerate() {
        if choice < count {
            return action;
        }
        choice = choice.saturating_sub(count);
    }
    visits
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(action, _)| action)
        .unwrap()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }
}
