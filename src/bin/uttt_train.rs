use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use az_nn::{configure_training_threads, AdamOptimizer, MlpNetwork, TrainingExample};
use az_pipeline::uttt_adapter::model_shape;
use az_pipeline::uttt_data::read_samples;

fn main() {
    let args: Vec<String> = env::args().collect();
    let data_paths = parse_data_paths(&args);
    let epochs = parse_arg(&args, "--epochs", 20);
    let optimizer_name = parse_string(&args, "--optimizer", "adam");
    let batch_size = parse_arg(&args, "--batch-size", 256).max(1);
    let threads = parse_arg(&args, "--threads", default_threads()).max(1);
    let default_learning_rate = if optimizer_name == "adam" {
        0.001
    } else {
        0.01
    };
    let learning_rate = parse_float_arg(&args, "--learning-rate", default_learning_rate);
    let output = optional_path(&args, "--output");
    let checkpoint = optional_path(&args, "--checkpoint");
    let optimizer_state = optional_path(&args, "--optimizer-state")
        .or_else(|| checkpoint.as_ref().map(|path| path.with_extension("adam")));
    assert!(epochs > 0, "--epochs must be greater than zero");
    assert!(
        learning_rate > 0.0,
        "--learning-rate must be greater than zero"
    );

    let mut samples = Vec::new();
    for path in &data_paths {
        samples.extend(read_samples(path).expect("read training data"));
    }
    assert!(!samples.is_empty(), "training data is empty");
    println!(
        "training: files={} samples={} epochs={} learning_rate={learning_rate}",
        data_paths.len(),
        samples.len(),
        epochs
    );
    io::stdout().flush().expect("flush training header");
    println!("training: optimizer={optimizer_name} batch_size={batch_size} threads={threads}");
    configure_training_threads(threads);

    let shape = model_shape();
    let model_loaded = checkpoint.as_deref().is_some_and(|path| path.exists());
    let mut network = match checkpoint.as_deref() {
        Some(path) if path.exists() => {
            println!("training: loading checkpoint {}", path.display());
            MlpNetwork::load(path, shape).expect("load model checkpoint")
        }
        _ => MlpNetwork::random(shape, 0x5eed),
    };
    let mut adam = match optimizer_name.as_str() {
        "adam" => Some(match optimizer_state.as_deref() {
            Some(path) if model_loaded && path.exists() => {
                println!("training: loading Adam state {}", path.display());
                AdamOptimizer::load(path, shape).expect("load Adam state")
            }
            _ => AdamOptimizer::new(shape),
        }),
        "sgd" => None,
        _ => panic!("unsupported optimizer {optimizer_name}; expected adam or sgd"),
    };
    let mut rng = Rng::new(0x5eed);
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let training_started = Instant::now();

    for epoch in 0..epochs {
        shuffle(&mut order, &mut rng);
        let mut loss = 0.0;
        let epoch_started = Instant::now();
        let progress_every = (order.len() / 10).max(1);
        let mut completed = 0;
        let mut next_progress = progress_every;
        for batch in order.chunks(batch_size) {
            let examples: Vec<_> = batch
                .iter()
                .map(|&index| {
                    let sample = &samples[index];
                    TrainingExample {
                        features: &sample.features,
                        policy: &sample.policy,
                        value: sample.value,
                    }
                })
                .collect();
            let batch_loss = if let Some(optimizer) = adam.as_mut() {
                network.train_batch_adam(&examples, learning_rate, optimizer)
            } else {
                network.train_batch(&examples, learning_rate)
            };
            loss += batch_loss * examples.len() as f32;
            completed += examples.len();
            if completed == order.len() || completed >= next_progress {
                println!(
                    "training: epoch={}/{} progress={:.0}% running_loss={:.6}",
                    epoch + 1,
                    epochs,
                    completed as f64 * 100.0 / order.len() as f64,
                    loss / completed as f32
                );
                io::stdout().flush().expect("flush training progress");
                next_progress += progress_every;
            }
        }
        println!(
            "training: epoch={} samples={} loss={:.6} elapsed={:.1}s",
            epoch + 1,
            samples.len(),
            loss / samples.len() as f32,
            epoch_started.elapsed().as_secs_f32()
        );
    }

    if let Some(path) = checkpoint.as_deref() {
        network.save(path).expect("save model checkpoint");
        println!("training: saved checkpoint {}", path.display());
    }
    if let (Some(optimizer), Some(path)) = (adam.as_ref(), optimizer_state.as_deref()) {
        optimizer.save(path).expect("save Adam state");
        println!(
            "training: saved Adam state {} at step {}",
            path.display(),
            optimizer.step()
        );
    }
    if let Some(path) = output.as_deref() {
        network
            .export_rust(path)
            .expect("export generated Rust weights");
        println!("training: exported {}", path.display());
    }
    println!(
        "training: total_elapsed={:.1}s",
        training_started.elapsed().as_secs_f32()
    );
}

fn parse_data_paths(args: &[String]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut index = 1;
    while index < args.len() {
        if args[index] == "--data" {
            let path = args.get(index + 1).expect("--data requires a path");
            paths.push(PathBuf::from(path));
            index += 2;
        } else {
            index += 1;
        }
    }
    if paths.is_empty() {
        paths.push(
            args.get(1)
                .filter(|value| !value.starts_with("--"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/uttt/selfplay.bin")),
        );
    }
    paths
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

fn optional_path(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn shuffle(values: &mut [usize], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        values.swap(index, rng.next_u32() as usize % (index + 1));
    }
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
