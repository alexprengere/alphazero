use std::collections::BTreeMap;
use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use az_core::Evaluator;
use az_mcts::{Mcts, RootNoise};
use az_nn::MlpNetwork;
use az_pipeline::connect4_adapter::{checkpoint_evaluator, heuristic_evaluator, model_shape};
use connect4_game::data::{SampleWriter, TrainingSample};
use connect4_game::features::{encode, FEATURE_SIZE};
use connect4_game::game::{Game, COLUMNS};

struct Sample {
    features: [u8; FEATURE_SIZE],
    policy: [f32; COLUMNS],
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
    let games = parse_arg(&args, "--games", 100);
    let simulations = parse_arg(&args, "--simulations", 200);
    let threads = parse_arg(&args, "--threads", default_threads()).max(1);
    let dirichlet_alpha = parse_float_arg(&args, "--dirichlet-alpha", 0.3);
    let dirichlet_epsilon = parse_float_arg(&args, "--dirichlet-epsilon", 0.25);
    let seed = parse_u64_arg(&args, "--seed", 0);
    let output = parse_path(&args, "--output", "data/connect4/selfplay.bin");
    let checkpoint = optional_path(&args, "--checkpoint");
    assert!(games > 0, "--games must be greater than zero");
    assert!(simulations > 0, "--simulations must be greater than zero");
    assert!(dirichlet_alpha > 0.0, "--dirichlet-alpha must be positive");
    assert!(
        (0.0..=1.0).contains(&dirichlet_epsilon),
        "--dirichlet-epsilon must be between zero and one"
    );
    let evaluator: Arc<dyn Evaluator<Game>> = if let Some(path) = checkpoint {
        checkpoint_evaluator(Arc::new(
            MlpNetwork::load(path, model_shape()).expect("load Connect Four checkpoint"),
        ))
    } else {
        heuristic_evaluator()
    };

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let temporary_output = temporary_path(&output);
    println!(
        "connect4 self-play: games={games} simulations={simulations} threads={threads} dirichlet={dirichlet_alpha}/{dirichlet_epsilon} seed={seed} evaluator={} output={}",
        if checkpoint.is_some() {
            "checkpoint"
        } else {
            "heuristic"
        },
        output.display()
    );
    io::stdout().flush().expect("flush header");

    let mut writer =
        SampleWriter::create(&temporary_output).expect("create temporary self-play output");
    let started = Instant::now();
    let progress_every = (games / 100).max(1);
    let next_game = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let worker_count = threads.min(games.max(1));
    let mut total_positions = 0;
    let mut outcomes = [0; 3];

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_game = Arc::clone(&next_game);
            let sender = sender.clone();
            let evaluator = Arc::clone(&evaluator);
            scope.spawn(move || loop {
                let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                if game_index >= games {
                    break;
                }
                let (samples, winner) = generate_game(
                    simulations,
                    evaluator.clone(),
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
                    .expect("send completed game");
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
                    "connect4 self-play: {completed}/{games} positions={total_positions} X/O/draw={}/{}/{} elapsed={} eta={}",
                    outcomes[0],
                    outcomes[1],
                    outcomes[2],
                    format_duration(elapsed),
                    format_duration(eta)
                );
            }
        }
        assert_eq!(completed, games, "self-play workers stopped early");
    });
    writer.flush().expect("flush self-play output");
    drop(writer);
    std::fs::rename(&temporary_output, &output).expect("publish self-play output");
}

fn generate_game(
    simulations: usize,
    evaluator: Arc<dyn Evaluator<Game>>,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    seed: u64,
) -> (Vec<Sample>, Option<i8>) {
    let mut game = Game::new();
    let mut search = Mcts::new(1.5, evaluator);
    let mut samples = Vec::with_capacity(42);
    let mut rng = Rng(seed);

    while !game.is_terminal() {
        let result = search.search_with_noise(
            &game,
            simulations,
            Some(RootNoise {
                alpha: dirichlet_alpha,
                epsilon: dirichlet_epsilon,
                seed: (seed << 32) ^ rng.next_u32() as u64,
            }),
        );
        let total = result
            .visits
            .iter()
            .map(|(_, count)| count)
            .sum::<u32>()
            .max(1) as f32;
        let mut policy = [0.0; COLUMNS];
        for (action, visits) in result.visits {
            policy[action] = visits as f32 / total;
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
        let action = if samples.len() <= 10 {
            sample_policy(&policy, &mut rng)
        } else {
            result.action
        };
        game.play(action);
        search.advance(action);
    }

    for sample in &mut samples {
        sample.value = game.outcome(sample.player);
    }
    (samples, game.winner())
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

fn sample_policy(policy: &[f32; COLUMNS], rng: &mut Rng) -> usize {
    let mut choice = rng.next_f32();
    for (action, probability) in policy.iter().enumerate() {
        if choice <= *probability {
            return action;
        }
        choice -= probability;
    }
    policy
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(action, _)| action)
        .unwrap()
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

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn optional_path<'a>(args: &'a [String], name: &str) -> Option<&'a Path> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(Path::new)
}

fn temporary_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .expect("output file name");
    output.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, usize::from)
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

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}
