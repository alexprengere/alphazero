use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

use az_nn::MlpNetwork;
use az_pipeline::connect4_adapter::model_shape;
use connect4_game::features::encode;
use connect4_game::game::{index, Game, COLUMNS, EMPTY, ROWS};

fn main() {
    let args: Vec<String> = env::args().collect();
    let human = parse_human_side(&args);
    let checkpoint = parse_path(&args, "--checkpoint", "models/connect4-current.bin");
    let network =
        MlpNetwork::load(&checkpoint, model_shape()).expect("load Connect Four checkpoint");
    let mut game = Game::new();
    let mut last_column = None;

    while !game.is_terminal() {
        render(&game, last_column);
        let column = if game.current_player() == human {
            read_human_column(&game)
        } else {
            choose_policy_column(&game, &network)
        };
        game.play(column);
        last_column = Some(column);
    }

    render(&game, last_column);
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
        .map(|side| if side.eq_ignore_ascii_case("O") { 1 } else { 0 })
        .unwrap_or(0)
}

fn choose_policy_column(game: &Game, network: &MlpNetwork) -> usize {
    let mut logits = [0.0; COLUMNS];
    network.evaluate(&encode(game), &mut logits);
    let column = (0..COLUMNS)
        .filter(|&column| game.is_legal(column))
        .max_by(|&left, &right| {
            let left_score = logits[left] + game.tactical_score(left);
            let right_score = logits[right] + game.tactical_score(right);
            left_score.total_cmp(&right_score)
        })
        .expect("policy called on terminal state");
    println!("Policy plays column {column}");
    column
}

fn read_human_column(game: &Game) -> usize {
    loop {
        print!("Your column (0-6, or q): ");
        io::stdout().flush().expect("flush prompt");
        let mut line = String::new();
        if io::stdin().read_line(&mut line).expect("read move") == 0
            || line.trim().eq_ignore_ascii_case("q")
        {
            println!("Game stopped.");
            std::process::exit(0);
        }
        let Ok(column) = line.trim().parse::<usize>() else {
            println!("Enter a column from 0 to 6.");
            continue;
        };
        if game.is_legal(column) {
            return column;
        }
        println!("That column is full or invalid.");
    }
}

fn render(game: &Game, last_column: Option<usize>) {
    println!("\n0 1 2 3 4 5 6");
    for row in (0..ROWS).rev() {
        for column in 0..COLUMNS {
            let owner = game.cells()[index(row, column)];
            let symbol = match owner {
                0 if Some(column) == last_column => 'x',
                1 if Some(column) == last_column => 'o',
                0 => 'X',
                1 => 'O',
                EMPTY => '.',
                _ => unreachable!(),
            };
            print!("{symbol} ");
        }
        println!();
    }
    println!();
}

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}
