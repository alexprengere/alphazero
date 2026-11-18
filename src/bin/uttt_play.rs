use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

use az_nn::MlpNetwork;
use az_pipeline::uttt_adapter::model_shape;
use az_pipeline::uttt_deploy::DeploymentState;
use az_pipeline::uttt_game::{action_to_row_col, Game, BOARD_CELLS, DRAW, EMPTY};
use az_pipeline::uttt_schema::action_from_row_col;

fn main() {
    let args: Vec<String> = env::args().collect();
    let human = parse_human_side(&args);
    let clear = !args.iter().any(|arg| arg == "--no-clear");
    let checkpoint = parse_path(&args, "--checkpoint", "models/uttt-current.bin");
    println!("Using checkpoint {}.", checkpoint.display());
    let policy = MlpNetwork::load(&checkpoint, model_shape()).expect("load UTTT checkpoint");
    let mut game = Game::new();
    let mut deployment = DeploymentState::new();
    let mut last_action = None;

    while !game.is_terminal() {
        render(&game, last_action, clear);
        let action = if game.current_player() == human {
            read_human_action(&game)
        } else {
            choose_policy_action(&game, &deployment, &policy)
        };
        game.play(action);
        deployment.record_move(action);
        last_action = Some(action);
    }

    render(&game, last_action, clear);
    match game.winner() {
        Some(winner) if winner == human => println!("You win."),
        Some(_) => println!("The policy wins."),
        None => println!("Draw."),
    }
}

fn parse_human_side(args: &[String]) -> i8 {
    args.iter()
        .position(|arg| arg == "--human")
        .and_then(|index| args.get(index + 1))
        .map(|side| side.to_ascii_uppercase())
        .map(|side| if side == "O" { 1 } else { 0 })
        .unwrap_or(0)
}

fn read_human_action(game: &Game) -> usize {
    loop {
        print!("Your move (row column, or q): ");
        io::stdout().flush().expect("flush prompt");
        let mut line = String::new();
        if io::stdin().read_line(&mut line).expect("read move") == 0 {
            println!("\nInput closed.");
            std::process::exit(0);
        }
        if line.trim().eq_ignore_ascii_case("q") {
            println!("Game stopped.");
            std::process::exit(0);
        }
        let mut values = line.split_whitespace().map(str::parse::<usize>);
        let Some(Ok(row)) = values.next() else {
            println!("Enter two numbers between 0 and 8.");
            continue;
        };
        let Some(Ok(col)) = values.next() else {
            println!("Enter two numbers between 0 and 8.");
            continue;
        };
        if row >= 9 || col >= 9 {
            println!("Coordinates must be between 0 and 8.");
            continue;
        }
        let action = action_from_row_col(row, col);
        if game.is_legal(action) {
            return action;
        }
        println!("That move is not legal in the currently forced micro-board.");
    }
}

fn choose_policy_action(game: &Game, deployment: &DeploymentState, policy: &MlpNetwork) -> usize {
    let mut legal = Vec::with_capacity(81);
    game.legal_actions(&mut legal);
    let features = deployment.encode(&legal);
    let mut logits = [0.0; BOARD_CELLS];
    policy.evaluate(&features, &mut logits);
    let action = legal
        .into_iter()
        .max_by(|&left, &right| {
            let left_score = logits[left] + deployment.fallback_score(left);
            let right_score = logits[right] + deployment.fallback_score(right);
            left_score.total_cmp(&right_score)
        })
        .expect("policy called on non-terminal state");
    let (row, col) = action_to_row_col(action);
    println!("Policy plays: {row} {col}");
    action
}

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn render(game: &Game, last_action: Option<usize>, clear: bool) {
    if clear {
        print!("\x1b[2J\x1b[H");
    }
    let next = if game.current_player() == 0 { 'X' } else { 'O' };
    println!("Ultimate Tic-Tac-Toe | next: {next}");
    match game.forced_board() {
        Some(board) => println!("Forced micro-board: {} {}", board / 3, board % 3),
        None => println!("Forced micro-board: any open board"),
    }
    println!("Coordinates are row column, from 0 to 8.\n");

    print!("  ");
    for col in 0..9 {
        if col == 3 || col == 6 {
            print!("  ");
        }
        print!("{col} ");
    }
    println!();

    for row in 0..9 {
        if row == 3 || row == 6 {
            println!("  ------+-------+------");
        }
        print!("{row} ");
        for col in 0..9 {
            if col == 3 || col == 6 {
                print!("| ");
            }
            let action = action_from_row_col(row, col);
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
    println!();
}
