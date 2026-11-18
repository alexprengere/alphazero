use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use az_nn::MlpNetwork;
use az_pipeline::uttt_adapter::{model_shape, EvaluatorKind, Mcts, RootNoise, SearchResult};
use az_pipeline::uttt_game::{action_to_row_col, Game, BOARD_CELLS, DRAW, EMPTY};

fn main() {
    let args: Vec<String> = env::args().collect();
    let simulations = parse_arg(&args, "--simulations", 200);
    let delay_ms = parse_arg(&args, "--delay-ms", 300);
    let seed = parse_arg(&args, "--seed", 1) as u64;
    let dirichlet_alpha = parse_float_arg(&args, "--dirichlet-alpha", 0.3);
    let dirichlet_epsilon = parse_float_arg(&args, "--dirichlet-epsilon", 0.25);
    let evaluator = if let Some(path) = optional_path(&args, "--checkpoint") {
        EvaluatorKind::Checkpoint(Arc::new(
            MlpNetwork::load(path, model_shape()).expect("load self-play checkpoint"),
        ))
    } else {
        EvaluatorKind::Heuristic
    };

    let mut game = Game::new();
    let mut search = Mcts::new(1.5, evaluator);
    let mut rng = Rng::new(seed);
    let mut last_action = None;
    let mut previous_search = None;
    let mut ply = 0;

    while !game.is_terminal() {
        render_frame(
            &game,
            last_action,
            previous_search.as_ref(),
            simulations,
            delay_ms,
        );

        let result = search.search_with_noise(
            &game,
            simulations,
            RootNoise {
                alpha: dirichlet_alpha,
                epsilon: dirichlet_epsilon,
                seed: (seed << 32) ^ rng.next_u32() as u64,
            },
        );
        let action = if ply < 20 {
            sample_visits(&result.visits, &mut rng)
        } else {
            result.action
        };
        game.play(action);
        search.advance(action);
        last_action = Some(action);
        previous_search = Some(result);
        ply += 1;
    }

    render_frame(&game, last_action, previous_search.as_ref(), simulations, 0);
    match game.winner() {
        Some(0) => println!("Result: X wins"),
        Some(1) => println!("Result: O wins"),
        None => println!("Result: draw"),
        _ => unreachable!(),
    }
}

fn render_frame(
    game: &Game,
    last_action: Option<usize>,
    previous_search: Option<&SearchResult>,
    simulations: usize,
    delay_ms: usize,
) {
    if delay_ms > 0 {
        print!("\x1b[2J\x1b[H");
    }
    let player = if game.current_player() == 0 { 'X' } else { 'O' };
    println!("Ultimate Tic-Tac-Toe self-play | next: {player} | MCTS: {simulations}");
    match game.forced_board() {
        Some(board) => println!("Forced micro-board: {} {}", board / 3, board % 3),
        None => println!("Forced micro-board: any open board"),
    }
    println!();

    for row in 0..9 {
        if row == 3 || row == 6 {
            println!("------+-------+------");
        }
        for col in 0..9 {
            if col == 3 || col == 6 {
                print!("| ");
            }
            let action = row * 9 + col;
            let symbol = match game.cells()[action] {
                0 if Some(action) == last_action => 'x',
                1 if Some(action) == last_action => 'o',
                0 => 'X',
                1 => 'O',
                _ => '.',
            };
            print!("{symbol} ");
        }
        println!();
    }

    println!("\nWon micro-boards:");
    for row in 0..3 {
        for col in 0..3 {
            let symbol = match game.micro_boards()[row * 3 + col] {
                0 => 'X',
                1 => 'O',
                DRAW => '=',
                EMPTY => '.',
                _ => unreachable!(),
            };
            print!("{symbol} ");
        }
        println!();
    }

    if let Some(result) = previous_search {
        let mut ranked: Vec<(usize, u32)> = result
            .visits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, visits)| *visits > 0)
            .collect();
        ranked.sort_unstable_by_key(|(_, visits)| std::cmp::Reverse(*visits));
        print!("\nPrevious search: ");
        for (index, (action, visits)) in ranked.into_iter().take(5).enumerate() {
            let (row, col) = action_to_row_col(action);
            if index > 0 {
                print!(", ");
            }
            print!("({row},{col})={visits}");
        }
        println!();
    }

    io::stdout().flush().expect("flush frame");
    if delay_ms > 0 {
        thread::sleep(Duration::from_millis(delay_ms as u64));
    }
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

fn optional_path<'a>(args: &'a [String], name: &str) -> Option<&'a Path> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(Path::new)
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
        .expect("non-empty visit distribution")
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
