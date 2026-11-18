use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use az_core::Game as GameContract;
use az_mcts::Mcts as GenericMcts;
use az_nn::MlpNetwork;
use az_pipeline::connect4_adapter;
use az_pipeline::uttt_adapter::{self, EvaluatorKind, Mcts as UtttMcts};
use connect4_game::Game as Connect4;
use uttt_game::game::Game as Uttt;

fn main() {
    let args: Vec<String> = env::args().collect();
    let game = parse_string(&args, "--game", "uttt");
    let champion = required_path(&args, "--champion");
    let candidate = required_path(&args, "--candidate");
    let games = parse_arg(&args, "--games", 100);
    let simulations = parse_arg(&args, "--simulations", 200);
    let threads = parse_arg(&args, "--threads", default_threads()).max(1);
    let opening_moves = parse_arg(&args, "--opening-moves", 4);
    let threshold = parse_float_arg(&args, "--threshold", 0.55);
    let result_path = optional_path(&args, "--result");
    assert!(
        games > 0 && games.is_multiple_of(2),
        "--games must be positive and even"
    );
    assert!(simulations > 0, "--simulations must be positive");
    assert!(
        (0.5..=1.0).contains(&threshold),
        "--threshold must be between 0.5 and 1.0"
    );
    let config = ArenaConfig {
        games,
        simulations,
        opening_moves,
        threads,
    };

    let (wins, draws, losses) = match game.as_str() {
        "uttt" => evaluate_uttt(&champion, &candidate, &config),
        "connect4" => evaluate_connect4(&champion, &candidate, &config),
        _ => panic!("unsupported game {game}; expected uttt or connect4"),
    };
    let score = (wins as f32 + 0.5 * draws as f32) / games as f32;
    println!(
        "arena: candidate W/D/L={wins}/{draws}/{losses} score={score:.4} threshold={threshold:.4}"
    );
    if let Some(path) = result_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create arena result directory");
        }
        fs::write(
            path,
            format!("{wins} {draws} {losses} {score:.8} {threshold:.8}\n"),
        )
        .expect("write arena result");
    }
    if score < threshold {
        std::process::exit(2);
    }
}

struct ArenaConfig {
    games: usize,
    simulations: usize,
    opening_moves: usize,
    threads: usize,
}

fn evaluate_uttt(
    champion_path: &Path,
    candidate_path: &Path,
    config: &ArenaConfig,
) -> (usize, usize, usize) {
    let shape = uttt_adapter::model_shape();
    let champion = EvaluatorKind::Checkpoint(Arc::new(
        MlpNetwork::load(champion_path, shape).expect("load UTTT champion"),
    ));
    let candidate = EvaluatorKind::Checkpoint(Arc::new(
        MlpNetwork::load(candidate_path, shape).expect("load UTTT candidate"),
    ));
    run_parallel_arena(config.games, config.threads, |game_index| {
        let candidate_player = (game_index % 2) as i8;
        let mut game = Uttt::new();
        let mut champion_search = UtttMcts::new(1.5, champion.clone());
        let mut candidate_search = UtttMcts::new(1.5, candidate.clone());
        let mut rng = Rng::new((game_index / 2 + 1) as u64);
        play_random_opening(&mut game, config.opening_moves, &mut rng, |action| {
            champion_search.advance(action);
            candidate_search.advance(action);
        });

        while !game.is_terminal() {
            let action = if game.current_player() == candidate_player {
                candidate_search.search(&game, config.simulations).action
            } else {
                champion_search.search(&game, config.simulations).action
            };
            game.play(action);
            champion_search.advance(action);
            candidate_search.advance(action);
        }
        game.outcome(candidate_player)
    })
}

fn evaluate_connect4(
    champion_path: &Path,
    candidate_path: &Path,
    config: &ArenaConfig,
) -> (usize, usize, usize) {
    let shape = connect4_adapter::model_shape();
    let champion = connect4_adapter::checkpoint_evaluator(Arc::new(
        MlpNetwork::load(champion_path, shape).expect("load Connect Four champion"),
    ));
    let candidate = connect4_adapter::checkpoint_evaluator(Arc::new(
        MlpNetwork::load(candidate_path, shape).expect("load Connect Four candidate"),
    ));
    run_parallel_arena(config.games, config.threads, |game_index| {
        let candidate_player = (game_index % 2) as i8;
        let mut game = Connect4::new();
        let mut champion_search = GenericMcts::new(1.5, Arc::clone(&champion));
        let mut candidate_search = GenericMcts::new(1.5, Arc::clone(&candidate));
        let mut rng = Rng::new((game_index / 2 + 1) as u64);
        play_random_opening(&mut game, config.opening_moves, &mut rng, |action| {
            champion_search.advance(action);
            candidate_search.advance(action);
        });

        while !game.is_terminal() {
            let action = if game.current_player() == candidate_player {
                candidate_search.search(&game, config.simulations).action
            } else {
                champion_search.search(&game, config.simulations).action
            };
            game.play(action);
            champion_search.advance(action);
            candidate_search.advance(action);
        }
        game.outcome(candidate_player)
    })
}

fn run_parallel_arena<F>(games: usize, threads: usize, play: F) -> (usize, usize, usize)
where
    F: Fn(usize) -> f32 + Sync,
{
    let next_game = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let worker_count = threads.min(games);
    let mut results = [0; 3];

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_game = Arc::clone(&next_game);
            let sender = sender.clone();
            let play = &play;
            scope.spawn(move || loop {
                let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                if game_index >= games {
                    break;
                }
                sender.send(play(game_index)).expect("send arena result");
            });
        }
        drop(sender);

        for (completed, outcome) in receiver.into_iter().enumerate() {
            record_result(outcome, &mut results);
            print_progress(completed + 1, games, results);
        }
    });
    assert_eq!(
        results.iter().sum::<usize>(),
        games,
        "arena workers stopped early"
    );
    (results[0], results[1], results[2])
}

fn play_random_opening<G, F>(game: &mut G, moves: usize, rng: &mut Rng, mut advance: F)
where
    G: GameContract<Action = usize>,
    F: FnMut(usize),
{
    let mut legal = Vec::new();
    for _ in 0..moves {
        if game.is_terminal() {
            break;
        }
        game.legal_actions(&mut legal);
        let action = legal[rng.next_u32() as usize % legal.len()];
        game.play(action);
        advance(action);
    }
}

fn record_result(outcome: f32, results: &mut [usize; 3]) {
    if outcome > 0.5 {
        results[0] += 1;
    } else if outcome < -0.5 {
        results[2] += 1;
    } else {
        results[1] += 1;
    }
}

fn print_progress(completed: usize, games: usize, results: [usize; 3]) {
    let every = (games / 10).max(1);
    if completed == games || completed.is_multiple_of(every) {
        println!(
            "arena: {completed}/{games} candidate W/D/L={}/{}/{}",
            results[0], results[1], results[2]
        );
    }
}

fn required_path(args: &[String], name: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} is required"))
}

fn optional_path(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
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

fn parse_string(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, usize::from)
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
