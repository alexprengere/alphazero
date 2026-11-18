use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use az_nn::MlpNetwork;
use az_pipeline::uttt_adapter::model_shape;

fn main() {
    let args: Vec<String> = env::args().collect();
    let output = parse_path(&args, "--output", "submission/uttt/main.rs");
    let checkpoint = parse_path(&args, "--checkpoint", "models/uttt-current.bin");
    let schema = read("crates/uttt-game/src/schema.rs");
    let deploy =
        read("crates/uttt-game/src/deploy.rs").replace("crate::schema", "crate::uttt_schema");
    let weights = MlpNetwork::load(&checkpoint, model_shape())
        .expect("load UTTT checkpoint")
        .export_rust_source();
    let network = read("templates/uttt_network.rs").replace(
        "mod weights {\n    include!(\"uttt_generated_weights.rs\");\n}",
        &format!("mod weights {{\n{weights}\n}}"),
    );
    let codingame = read("templates/uttt_codingame.rs").replace("az_pipeline::", "crate::");

    let source = format!(
        "mod uttt_schema {{\n{schema}\n}}\n\
         mod uttt_deploy {{\n{deploy}\n}}\n\
         mod uttt_network {{\n{network}\n}}\n\
         {codingame}\n"
    );
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("create submission directory");
    }
    fs::write(&output, source).expect("write submission");
    eprintln!(
        "wrote {} bytes to {}",
        fs::metadata(&output).expect("submission metadata").len(),
        output.display()
    );
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path).expect("read source module")
}

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}
