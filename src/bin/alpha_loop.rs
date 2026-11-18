use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let game = parse_string(&args, "--game", "uttt");
    let config = match game.as_str() {
        "uttt" => GameConfig {
            selfplay_bin: "uttt-selfplay",
            train_bin: "uttt-train",
            data_dir: "data/uttt/iterations",
            checkpoint: "models/uttt-current.bin",
        },
        "connect4" => GameConfig {
            selfplay_bin: "connect4-selfplay",
            train_bin: "connect4-train",
            data_dir: "data/connect4/iterations",
            checkpoint: "models/connect4-current.bin",
        },
        _ => panic!("unsupported game {game}; expected uttt or connect4"),
    };
    let iterations = parse_arg(&args, "--iterations", 10);
    let games = parse_arg(&args, "--games", 100);
    let simulations = parse_arg(&args, "--simulations", 400);
    let epochs = parse_arg(&args, "--epochs", 5);
    let optimizer = parse_string(&args, "--optimizer", "adam");
    let batch_size = parse_arg(&args, "--batch-size", 256).max(1);
    let replay_window = parse_arg(&args, "--replay-window", 5);
    let threads = parse_arg(&args, "--threads", default_threads()).max(1);
    let dirichlet_alpha = parse_float_arg(&args, "--dirichlet-alpha", 0.3);
    let dirichlet_epsilon = parse_float_arg(&args, "--dirichlet-epsilon", 0.25);
    let arena_games = parse_arg(&args, "--arena-games", 100);
    let arena_simulations = parse_arg(&args, "--arena-simulations", simulations);
    let arena_opening_moves = parse_arg(&args, "--arena-opening-moves", 4);
    let promotion_threshold = parse_float_arg(&args, "--promotion-threshold", 0.55);
    let default_learning_rate = if optimizer == "adam" { 0.001 } else { 0.01 };
    let learning_rate = parse_float_arg(&args, "--learning-rate", default_learning_rate);
    let data_dir = parse_path(&args, "--data-dir", config.data_dir);
    let checkpoint = parse_path(&args, "--checkpoint", config.checkpoint);
    assert!(iterations > 0, "--iterations must be greater than zero");
    assert!(games > 0, "--games must be greater than zero");
    assert!(simulations > 0, "--simulations must be greater than zero");
    assert!(epochs > 0, "--epochs must be greater than zero");
    assert!(
        arena_games > 0 && arena_games.is_multiple_of(2),
        "--arena-games must be positive and even"
    );
    assert!(
        replay_window > 0,
        "--replay-window must be greater than zero"
    );
    assert!(
        learning_rate > 0.0,
        "--learning-rate must be greater than zero"
    );
    assert!(dirichlet_alpha > 0.0, "--dirichlet-alpha must be positive");
    assert!(
        (0.0..=1.0).contains(&dirichlet_epsilon),
        "--dirichlet-epsilon must be between zero and one"
    );

    fs::create_dir_all(&data_dir).expect("create iteration data directory");
    if let Some(parent) = checkpoint.parent() {
        fs::create_dir_all(parent).expect("create model directory");
    }

    println!(
        "alpha-loop: game={game} iterations={iterations} games={games} simulations={simulations} threads={threads} epochs={epochs} optimizer={optimizer} batch_size={batch_size} replay_window={replay_window} learning_rate={learning_rate}"
    );
    println!(
        "alpha-loop: arena_games={arena_games} arena_simulations={arena_simulations} promotion_threshold={promotion_threshold}"
    );
    println!("alpha-loop: self-play Dirichlet alpha={dirichlet_alpha} epsilon={dirichlet_epsilon}");
    println!(
        "alpha-loop: data_dir={} checkpoint={}",
        data_dir.display(),
        checkpoint.display()
    );
    io::stdout().flush().expect("flush loop header");

    run(Command::new("cargo").args([
        "build",
        "--release",
        "--bin",
        config.train_bin,
        "--bin",
        config.selfplay_bin,
        "--bin",
        "arena",
    ]));

    let mut shards = existing_shards(&data_dir);
    let first_iteration = next_iteration(&shards);
    let mut model_available = checkpoint.exists();
    let candidate = candidate_path(&checkpoint);
    let champion_optimizer = checkpoint.with_extension("adam");
    let candidate_optimizer = candidate.with_extension("adam");
    let mut summaries = Vec::with_capacity(iterations);

    for (round, iteration) in (first_iteration..first_iteration + iterations).enumerate() {
        println!(
            "\n=== AlphaZero iteration {} of {} ===",
            round + 1,
            iterations
        );
        println!(
            "iteration {iteration}: self-play evaluator={}",
            if model_available {
                "runtime checkpoint"
            } else {
                "heuristic bootstrap"
            }
        );
        io::stdout().flush().expect("flush iteration header");

        let shard = data_dir.join(format!("iteration-{iteration:04}.bin"));
        let mut selfplay = Command::new(format!("target/release/{}", config.selfplay_bin));
        selfplay
            .arg("--games")
            .arg(games.to_string())
            .arg("--simulations")
            .arg(simulations.to_string())
            .arg("--threads")
            .arg(threads.to_string())
            .arg("--dirichlet-alpha")
            .arg(dirichlet_alpha.to_string())
            .arg("--dirichlet-epsilon")
            .arg(dirichlet_epsilon.to_string())
            .arg("--seed")
            .arg((iteration as u64 + 1).to_string())
            .arg("--output")
            .arg(&shard);
        if model_available {
            selfplay.arg("--checkpoint").arg(&checkpoint);
        }
        run(&mut selfplay);
        shards.push(shard);

        let window_start = shards.len().saturating_sub(replay_window);
        println!(
            "iteration {iteration}: training with {} replay shard(s)",
            shards.len() - window_start
        );
        prepare_candidate(
            &checkpoint,
            &candidate,
            &champion_optimizer,
            &candidate_optimizer,
            model_available,
            &optimizer,
        );
        let mut train = Command::new(format!("target/release/{}", config.train_bin));
        for path in &shards[window_start..] {
            train.arg("--data").arg(path);
        }
        train
            .arg("--epochs")
            .arg(epochs.to_string())
            .arg("--learning-rate")
            .arg(learning_rate.to_string())
            .arg("--optimizer")
            .arg(&optimizer)
            .arg("--batch-size")
            .arg(batch_size.to_string())
            .arg("--threads")
            .arg(threads.to_string())
            .arg("--checkpoint")
            .arg(&candidate);
        run(&mut train);

        let arena_result_path = candidate.with_extension("arena-result");
        remove_if_exists(&arena_result_path);
        let arena_outcome = if model_available {
            let mut arena = Command::new("target/release/arena");
            arena
                .arg("--game")
                .arg(&game)
                .arg("--champion")
                .arg(&checkpoint)
                .arg("--candidate")
                .arg(&candidate)
                .arg("--games")
                .arg(arena_games.to_string())
                .arg("--simulations")
                .arg(arena_simulations.to_string())
                .arg("--threads")
                .arg(threads.to_string())
                .arg("--opening-moves")
                .arg(arena_opening_moves.to_string())
                .arg("--threshold")
                .arg(promotion_threshold.to_string())
                .arg("--result")
                .arg(&arena_result_path);
            let promoted = match execute(&mut arena).code() {
                Some(0) => true,
                Some(2) => false,
                code => panic!("arena failed with exit code {code:?}"),
            };
            let outcome = read_arena_result(&arena_result_path);
            remove_if_exists(&arena_result_path);
            Some((promoted, outcome))
        } else {
            None
        };
        let promoted = arena_outcome.as_ref().is_none_or(|(promoted, _)| *promoted);

        if promoted {
            promote_candidate(
                &candidate,
                &checkpoint,
                &candidate_optimizer,
                &champion_optimizer,
                &optimizer,
            );
        } else {
            remove_if_exists(&candidate);
            remove_if_exists(&candidate_optimizer);
        }
        print_promotion_result(iteration, promoted, arena_outcome.as_ref());
        summaries.push(IterationSummary {
            iteration,
            promoted,
            arena: arena_outcome.as_ref().map(|(_, result)| *result),
        });
        model_available = true;
    }

    print_training_summary(&summaries);
    println!(
        "\nCompleted {iterations} iterations. Latest checkpoint: {}",
        checkpoint.display()
    );
    if game == "uttt" {
        println!("Run `cargo run --release --bin uttt-package` to generate the submission.");
    } else {
        println!("Run `cargo run --release --bin connect4-play` to play the checkpoint.");
    }
}

struct GameConfig {
    selfplay_bin: &'static str,
    train_bin: &'static str,
    data_dir: &'static str,
    checkpoint: &'static str,
}

#[derive(Clone, Copy)]
struct ArenaOutcome {
    wins: usize,
    draws: usize,
    losses: usize,
    score: f32,
    threshold: f32,
}

struct IterationSummary {
    iteration: usize,
    promoted: bool,
    arena: Option<ArenaOutcome>,
}

fn run(command: &mut Command) {
    let status = execute(command);
    let program = format!("{command:?}");
    require_success(status, &program);
}

fn execute(command: &mut Command) -> ExitStatus {
    let program = format!("{command:?}");
    println!("running: {program}");
    io::stdout().flush().expect("flush command");
    let started = Instant::now();
    let status = command.status().expect("start child command");
    println!(
        "completed in {:.1}s with {status}: {program}",
        started.elapsed().as_secs_f32(),
    );
    status
}

fn require_success(status: ExitStatus, program: &str) {
    if !status.success() {
        panic!("command failed with {status}: {program}");
    }
}

fn existing_shards(data_dir: &Path) -> Vec<PathBuf> {
    let mut shards: Vec<PathBuf> = fs::read_dir(data_dir)
        .expect("read iteration data directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("iteration-") && name.ends_with(".bin"))
        })
        .collect();
    shards.sort();
    shards
}

fn next_iteration(shards: &[PathBuf]) -> usize {
    shards
        .iter()
        .filter_map(|path| path.file_stem()?.to_str()?.strip_prefix("iteration-"))
        .filter_map(|number| number.parse::<usize>().ok())
        .max()
        .map_or(0, |number| number + 1)
}

fn candidate_path(champion: &Path) -> PathBuf {
    let stem = champion
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("checkpoint file stem");
    let extension = champion
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("bin");
    champion.with_file_name(format!("{stem}-candidate.{extension}"))
}

fn prepare_candidate(
    champion: &Path,
    candidate: &Path,
    champion_optimizer: &Path,
    candidate_optimizer: &Path,
    model_available: bool,
    optimizer: &str,
) {
    if model_available {
        fs::copy(champion, candidate).expect("copy champion to candidate");
        if optimizer == "adam" && champion_optimizer.exists() {
            fs::copy(champion_optimizer, candidate_optimizer).expect("copy champion Adam state");
        } else {
            remove_if_exists(candidate_optimizer);
        }
    } else {
        remove_if_exists(candidate);
        remove_if_exists(candidate_optimizer);
    }
}

fn promote_candidate(
    candidate: &Path,
    champion: &Path,
    candidate_optimizer: &Path,
    champion_optimizer: &Path,
    optimizer: &str,
) {
    replace_file(candidate, champion);
    if optimizer == "adam" && candidate_optimizer.exists() {
        replace_file(candidate_optimizer, champion_optimizer);
    } else {
        remove_if_exists(champion_optimizer);
    }
}

fn replace_file(source: &Path, target: &Path) {
    let temporary = target.with_extension("promote");
    fs::copy(source, &temporary).expect("copy promoted checkpoint");
    fs::rename(&temporary, target).expect("replace champion checkpoint");
    remove_if_exists(source);
}

fn remove_if_exists(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("remove {}: {error}", path.display()),
    }
}

fn read_arena_result(path: &Path) -> ArenaOutcome {
    let content = fs::read_to_string(path).expect("read arena result");
    let mut values = content.split_whitespace();
    ArenaOutcome {
        wins: values.next().unwrap().parse().unwrap(),
        draws: values.next().unwrap().parse().unwrap(),
        losses: values.next().unwrap().parse().unwrap(),
        score: values.next().unwrap().parse().unwrap(),
        threshold: values.next().unwrap().parse().unwrap(),
    }
}

fn print_promotion_result(iteration: usize, promoted: bool, arena: Option<&(bool, ArenaOutcome)>) {
    println!("\n========================================================================");
    if let Some((_, result)) = arena {
        println!(
            "CANDIDATE {} | ITERATION {iteration}",
            if promoted { "PROMOTED" } else { "REJECTED" }
        );
        println!(
            "Arena score: {:.4} {} threshold {:.4}",
            result.score,
            if promoted { ">=" } else { "<" },
            result.threshold
        );
        println!(
            "Candidate record: {} wins / {} draws / {} losses",
            result.wins, result.draws, result.losses
        );
    } else {
        println!("INITIAL CHAMPION CREATED | ITERATION {iteration}");
        println!("Arena skipped because no previous champion existed.");
    }
    println!("========================================================================\n");
}

fn print_training_summary(summaries: &[IterationSummary]) {
    println!("\n====================== TRAINING PROMOTION SUMMARY ======================");
    println!("| Iteration | Decision  | Wins | Draws | Losses | Score  |");
    println!("|-----------|-----------|------|-------|--------|--------|");
    for summary in summaries {
        if let Some(arena) = summary.arena {
            println!(
                "| {:>9} | {:<9} | {:>4} | {:>5} | {:>6} | {:>6.3} |",
                summary.iteration,
                if summary.promoted {
                    "PROMOTED"
                } else {
                    "REJECTED"
                },
                arena.wins,
                arena.draws,
                arena.losses,
                arena.score
            );
        } else {
            println!(
                "| {:>9} | {:<9} | {:>4} | {:>5} | {:>6} | {:>6} |",
                summary.iteration, "INITIAL", "-", "-", "-", "-"
            );
        }
    }

    let evaluated: Vec<_> = summaries
        .iter()
        .filter_map(|summary| summary.arena.map(|arena| (summary, arena)))
        .collect();
    let promoted = evaluated
        .iter()
        .filter(|(summary, _)| summary.promoted)
        .count();
    let rejected = evaluated.len() - promoted;
    let average_score = if evaluated.is_empty() {
        0.0
    } else {
        evaluated.iter().map(|(_, arena)| arena.score).sum::<f32>() / evaluated.len() as f32
    };
    let promotion_rate = if evaluated.is_empty() {
        0.0
    } else {
        promoted as f32 * 100.0 / evaluated.len() as f32
    };
    println!("------------------------------------------------------------------------");
    println!(
        "Evaluated candidates: {} | Promoted: {} | Rejected: {}",
        evaluated.len(),
        promoted,
        rejected
    );
    println!(
        "Promotion rate: {:.1}% | Average candidate score: {:.4}",
        promotion_rate, average_score
    );
    println!("========================================================================\n");
}

fn parse_arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
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

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}
