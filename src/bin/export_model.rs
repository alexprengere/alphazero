use std::env;
use std::path::PathBuf;

use az_nn::MlpNetwork;
use az_pipeline::{connect4_adapter, uttt_adapter};

fn main() {
    let args: Vec<String> = env::args().collect();
    let game = parse_string(&args, "--game", "uttt");
    let (shape, default_checkpoint, default_output) = match game.as_str() {
        "uttt" => (
            uttt_adapter::model_shape(),
            "models/uttt-current.bin",
            "models/uttt-generated-weights.rs",
        ),
        "connect4" => (
            connect4_adapter::model_shape(),
            "models/connect4-current.bin",
            "models/connect4-generated-weights.rs",
        ),
        _ => panic!("unsupported game {game}; expected uttt or connect4"),
    };
    let checkpoint = parse_path(&args, "--checkpoint", default_checkpoint);
    let output = parse_path(&args, "--output", default_output);

    let network = MlpNetwork::load(&checkpoint, shape).expect("load model checkpoint");
    network.export_rust(&output).expect("export Rust weights");
    println!(
        "exported {} checkpoint {} to {}",
        game,
        checkpoint.display(),
        output.display()
    );
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
