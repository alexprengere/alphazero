use std::cell::RefCell;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;

const LEGACY_CHECKPOINT_MAGIC: &[u8; 8] = b"UTTTNN1\n";
const CHECKPOINT_MAGIC: &[u8; 8] = b"AZMLP002";
const ADAM_MAGIC: &[u8; 8] = b"AZADAM01";

thread_local! {
    static INFERENCE_HIDDEN: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
pub struct MlpShape {
    pub input: usize,
    pub hidden: usize,
    pub policy: usize,
}

#[derive(Clone)]
pub struct MlpNetwork {
    shape: MlpShape,
    input_weights: Vec<f32>,
    input_bias: Vec<f32>,
    policy_weights: Vec<f32>,
    policy_bias: Vec<f32>,
    value_weights: Vec<f32>,
    value_bias: f32,
}

pub struct TrainingWorkspace {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
    hidden_gradient: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct TrainingExample<'a> {
    pub features: &'a [u8],
    pub policy: &'a [f32],
    pub value: f32,
}

struct BatchGradient {
    input_weights: Vec<f32>,
    input_bias: Vec<f32>,
    policy_weights: Vec<f32>,
    policy_bias: Vec<f32>,
    value_weights: Vec<f32>,
    value_bias: f32,
    loss: f32,
    samples: usize,
}

struct ParameterSet {
    input_weights: Vec<f32>,
    input_bias: Vec<f32>,
    policy_weights: Vec<f32>,
    policy_bias: Vec<f32>,
    value_weights: Vec<f32>,
    value_bias: f32,
}

#[derive(Clone, Copy)]
struct AdamUpdate {
    gradient_scale: f32,
    learning_rate: f32,
    correction1: f32,
    correction2: f32,
}

pub struct AdamOptimizer {
    shape: MlpShape,
    step: u64,
    first_moment: ParameterSet,
    second_moment: ParameterSet,
}

impl ParameterSet {
    fn zeros(shape: MlpShape) -> Self {
        Self {
            input_weights: vec![0.0; shape.hidden * shape.input],
            input_bias: vec![0.0; shape.hidden],
            policy_weights: vec![0.0; shape.policy * shape.hidden],
            policy_bias: vec![0.0; shape.policy],
            value_weights: vec![0.0; shape.hidden],
            value_bias: 0.0,
        }
    }
}

impl AdamOptimizer {
    pub fn new(shape: MlpShape) -> Self {
        Self {
            shape,
            step: 0,
            first_moment: ParameterSet::zeros(shape),
            second_moment: ParameterSet::zeros(shape),
        }
    }

    pub fn load(path: &Path, shape: MlpShape) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0; ADAM_MAGIC.len()];
        input.read_exact(&mut magic)?;
        if &magic != ADAM_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid Adam checkpoint",
            ));
        }
        let stored_shape = MlpShape {
            input: read_u32(&mut input)? as usize,
            hidden: read_u32(&mut input)? as usize,
            policy: read_u32(&mut input)? as usize,
        };
        if stored_shape.input != shape.input
            || stored_shape.hidden != shape.hidden
            || stored_shape.policy != shape.policy
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Adam checkpoint shape does not match model",
            ));
        }
        let mut step_bytes = [0; 8];
        input.read_exact(&mut step_bytes)?;
        let parameter_count = shape.hidden * shape.input
            + shape.hidden
            + shape.policy * shape.hidden
            + shape.policy
            + shape.hidden
            + 1;
        let expected_size = 28 + parameter_count as u64 * 8;
        if file_size != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid Adam checkpoint size",
            ));
        }

        let mut optimizer = Self::new(shape);
        optimizer.step = u64::from_le_bytes(step_bytes);
        read_parameter_set(&mut input, &mut optimizer.first_moment)?;
        read_parameter_set(&mut input, &mut optimizer.second_moment)?;
        Ok(optimizer)
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(ADAM_MAGIC)?;
        output.write_all(&(self.shape.input as u32).to_le_bytes())?;
        output.write_all(&(self.shape.hidden as u32).to_le_bytes())?;
        output.write_all(&(self.shape.policy as u32).to_le_bytes())?;
        output.write_all(&self.step.to_le_bytes())?;
        write_parameter_set(&mut output, &self.first_moment)?;
        write_parameter_set(&mut output, &self.second_moment)?;
        output.flush()
    }

    pub fn step(&self) -> u64 {
        self.step
    }
}

impl BatchGradient {
    fn new(shape: MlpShape) -> Self {
        Self {
            input_weights: vec![0.0; shape.hidden * shape.input],
            input_bias: vec![0.0; shape.hidden],
            policy_weights: vec![0.0; shape.policy * shape.hidden],
            policy_bias: vec![0.0; shape.policy],
            value_weights: vec![0.0; shape.hidden],
            value_bias: 0.0,
            loss: 0.0,
            samples: 0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        add_slices(&mut self.input_weights, &other.input_weights);
        add_slices(&mut self.input_bias, &other.input_bias);
        add_slices(&mut self.policy_weights, &other.policy_weights);
        add_slices(&mut self.policy_bias, &other.policy_bias);
        add_slices(&mut self.value_weights, &other.value_weights);
        self.value_bias += other.value_bias;
        self.loss += other.loss;
        self.samples += other.samples;
        self
    }
}

pub fn configure_training_threads(threads: usize) {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build_global();
}

impl TrainingWorkspace {
    pub fn new(shape: MlpShape) -> Self {
        Self {
            hidden: vec![0.0; shape.hidden],
            logits: vec![0.0; shape.policy],
            probabilities: vec![0.0; shape.policy],
            hidden_gradient: vec![0.0; shape.hidden],
        }
    }
}

impl MlpNetwork {
    pub fn random(shape: MlpShape, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut input_weights = vec![0.0; shape.hidden * shape.input];
        let mut policy_weights = vec![0.0; shape.policy * shape.hidden];
        for weight in &mut input_weights {
            *weight = rng.symmetric(0.08);
        }
        for weight in &mut policy_weights {
            *weight = rng.symmetric(0.08);
        }
        let mut value_weights = vec![0.0; shape.hidden];
        for weight in &mut value_weights {
            *weight = rng.symmetric(0.08);
        }
        Self {
            shape,
            input_weights,
            input_bias: vec![0.0; shape.hidden],
            policy_weights,
            policy_bias: vec![0.0; shape.policy],
            value_weights,
            value_bias: 0.0,
        }
    }

    pub fn load(path: &Path, shape: MlpShape) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0; CHECKPOINT_MAGIC.len()];
        input.read_exact(&mut magic)?;
        let header_size = if &magic == CHECKPOINT_MAGIC {
            let stored_shape = MlpShape {
                input: read_u32(&mut input)? as usize,
                hidden: read_u32(&mut input)? as usize,
                policy: read_u32(&mut input)? as usize,
            };
            if stored_shape.input != shape.input
                || stored_shape.hidden != shape.hidden
                || stored_shape.policy != shape.policy
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "checkpoint shape is {}x{}x{}, expected {}x{}x{}",
                        stored_shape.input,
                        stored_shape.hidden,
                        stored_shape.policy,
                        shape.input,
                        shape.hidden,
                        shape.policy
                    ),
                ));
            }
            20
        } else if &magic != LEGACY_CHECKPOINT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid model checkpoint",
            ));
        } else {
            8
        };
        let parameter_count = shape.hidden * shape.input
            + shape.hidden
            + shape.policy * shape.hidden
            + shape.policy
            + shape.hidden
            + 1;
        let expected_size = header_size + parameter_count as u64 * 4;
        if file_size != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "checkpoint has {file_size} bytes, expected {expected_size} for this model"
                ),
            ));
        }

        let mut network = Self {
            shape,
            input_weights: vec![0.0; shape.hidden * shape.input],
            input_bias: vec![0.0; shape.hidden],
            policy_weights: vec![0.0; shape.policy * shape.hidden],
            policy_bias: vec![0.0; shape.policy],
            value_weights: vec![0.0; shape.hidden],
            value_bias: 0.0,
        };
        read_f32s(&mut input, &mut network.input_weights)?;
        read_f32s(&mut input, &mut network.input_bias)?;
        read_f32s(&mut input, &mut network.policy_weights)?;
        read_f32s(&mut input, &mut network.policy_bias)?;
        read_f32s(&mut input, &mut network.value_weights)?;
        let mut value_bias = [0; 4];
        input.read_exact(&mut value_bias)?;
        network.value_bias = f32::from_le_bytes(value_bias);
        Ok(network)
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(CHECKPOINT_MAGIC)?;
        output.write_all(&(self.shape.input as u32).to_le_bytes())?;
        output.write_all(&(self.shape.hidden as u32).to_le_bytes())?;
        output.write_all(&(self.shape.policy as u32).to_le_bytes())?;
        write_f32s(&mut output, &self.input_weights)?;
        write_f32s(&mut output, &self.input_bias)?;
        write_f32s(&mut output, &self.policy_weights)?;
        write_f32s(&mut output, &self.policy_bias)?;
        write_f32s(&mut output, &self.value_weights)?;
        output.write_all(&self.value_bias.to_le_bytes())?;
        output.flush()
    }

    pub fn evaluate(&self, features: &[f32], policy: &mut [f32]) -> f32 {
        assert_eq!(features.len(), self.shape.input);
        assert_eq!(policy.len(), self.shape.policy);
        INFERENCE_HIDDEN.with(|hidden| {
            let mut hidden = hidden.borrow_mut();
            hidden.resize(self.shape.hidden, 0.0);
            self.forward_hidden(features, &mut hidden);
            self.forward_heads(&hidden, policy)
        })
    }

    pub fn train_sample(
        &mut self,
        features: &[u8],
        target_policy: &[f32],
        target_value: f32,
        learning_rate: f32,
        workspace: &mut TrainingWorkspace,
    ) -> f32 {
        assert_eq!(features.len(), self.shape.input);
        assert_eq!(target_policy.len(), self.shape.policy);
        workspace.hidden.fill(0.0);
        for output in 0..self.shape.hidden {
            let offset = output * self.shape.input;
            let mut sum = self.input_bias[output];
            for input in 0..self.shape.input {
                sum += features[input] as f32 * self.input_weights[offset + input];
            }
            workspace.hidden[output] = sum.max(0.0);
        }

        for output in 0..self.shape.policy {
            let offset = output * self.shape.hidden;
            let mut sum = self.policy_bias[output];
            for input in 0..self.shape.hidden {
                sum += workspace.hidden[input] * self.policy_weights[offset + input];
            }
            workspace.logits[output] = sum;
        }
        let maximum = workspace
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut normalizer = 0.0;
        for action in 0..self.shape.policy {
            workspace.probabilities[action] = (workspace.logits[action] - maximum).exp();
            normalizer += workspace.probabilities[action];
        }
        for probability in &mut workspace.probabilities {
            *probability /= normalizer;
        }

        let value_logit = self
            .value_weights
            .iter()
            .zip(&workspace.hidden)
            .fold(self.value_bias, |sum, (weight, activation)| {
                sum + weight * activation
            });
        let value = value_logit.tanh();

        workspace.hidden_gradient.fill(0.0);
        for action in 0..self.shape.policy {
            let gradient = workspace.probabilities[action] - target_policy[action];
            let offset = action * self.shape.hidden;
            for hidden_index in 0..self.shape.hidden {
                workspace.hidden_gradient[hidden_index] +=
                    gradient * self.policy_weights[offset + hidden_index];
            }
        }
        let value_gradient = 2.0 * (value - target_value) * (1.0 - value * value);
        for hidden_index in 0..self.shape.hidden {
            workspace.hidden_gradient[hidden_index] +=
                value_gradient * self.value_weights[hidden_index];
            if workspace.hidden[hidden_index] == 0.0 {
                workspace.hidden_gradient[hidden_index] = 0.0;
            }
        }

        for action in 0..self.shape.policy {
            let gradient = workspace.probabilities[action] - target_policy[action];
            let offset = action * self.shape.hidden;
            for hidden_index in 0..self.shape.hidden {
                self.policy_weights[offset + hidden_index] -=
                    learning_rate * gradient * workspace.hidden[hidden_index];
            }
            self.policy_bias[action] -= learning_rate * gradient;
        }
        for hidden_index in 0..self.shape.hidden {
            self.value_weights[hidden_index] -=
                learning_rate * value_gradient * workspace.hidden[hidden_index];
        }
        self.value_bias -= learning_rate * value_gradient;

        for hidden_index in 0..self.shape.hidden {
            let gradient = workspace.hidden_gradient[hidden_index];
            let offset = hidden_index * self.shape.input;
            for input in 0..self.shape.input {
                if features[input] != 0 {
                    self.input_weights[offset + input] -= learning_rate * gradient;
                }
            }
            self.input_bias[hidden_index] -= learning_rate * gradient;
        }

        let policy_loss = target_policy
            .iter()
            .zip(&workspace.probabilities)
            .filter(|(target, _)| **target > 0.0)
            .map(|(target, probability)| -target * probability.max(1e-12).ln())
            .sum::<f32>();
        policy_loss + (value - target_value).powi(2)
    }

    pub fn train_batch(&mut self, examples: &[TrainingExample<'_>], learning_rate: f32) -> f32 {
        let gradient = self.batch_gradient(examples);
        let scale = learning_rate / gradient.samples as f32;
        subtract_scaled(&mut self.input_weights, &gradient.input_weights, scale);
        subtract_scaled(&mut self.input_bias, &gradient.input_bias, scale);
        subtract_scaled(&mut self.policy_weights, &gradient.policy_weights, scale);
        subtract_scaled(&mut self.policy_bias, &gradient.policy_bias, scale);
        subtract_scaled(&mut self.value_weights, &gradient.value_weights, scale);
        self.value_bias -= scale * gradient.value_bias;
        gradient.loss / gradient.samples as f32
    }

    pub fn train_batch_adam(
        &mut self,
        examples: &[TrainingExample<'_>],
        learning_rate: f32,
        optimizer: &mut AdamOptimizer,
    ) -> f32 {
        assert_eq!(optimizer.shape.input, self.shape.input);
        assert_eq!(optimizer.shape.hidden, self.shape.hidden);
        assert_eq!(optimizer.shape.policy, self.shape.policy);
        let gradient = self.batch_gradient(examples);
        optimizer.step += 1;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let correction1 = 1.0 - beta1.powf(optimizer.step as f32);
        let correction2 = 1.0 - beta2.powf(optimizer.step as f32);
        let gradient_scale = 1.0 / gradient.samples as f32;
        let update = AdamUpdate {
            gradient_scale,
            learning_rate,
            correction1,
            correction2,
        };

        adam_update_slice(
            &mut self.input_weights,
            &mut optimizer.first_moment.input_weights,
            &mut optimizer.second_moment.input_weights,
            &gradient.input_weights,
            update,
        );
        adam_update_slice(
            &mut self.input_bias,
            &mut optimizer.first_moment.input_bias,
            &mut optimizer.second_moment.input_bias,
            &gradient.input_bias,
            update,
        );
        adam_update_slice(
            &mut self.policy_weights,
            &mut optimizer.first_moment.policy_weights,
            &mut optimizer.second_moment.policy_weights,
            &gradient.policy_weights,
            update,
        );
        adam_update_slice(
            &mut self.policy_bias,
            &mut optimizer.first_moment.policy_bias,
            &mut optimizer.second_moment.policy_bias,
            &gradient.policy_bias,
            update,
        );
        adam_update_slice(
            &mut self.value_weights,
            &mut optimizer.first_moment.value_weights,
            &mut optimizer.second_moment.value_weights,
            &gradient.value_weights,
            update,
        );
        adam_update_scalar(
            &mut self.value_bias,
            &mut optimizer.first_moment.value_bias,
            &mut optimizer.second_moment.value_bias,
            gradient.value_bias * gradient_scale,
            update,
        );
        gradient.loss / gradient.samples as f32
    }

    pub fn export_rust(&self, path: &Path) -> io::Result<()> {
        let output = self.export_rust_source();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, output)
    }

    pub fn export_rust_source(&self) -> String {
        let (input_weights, input_scale) = quantize(&self.input_weights);
        let (policy_weights, policy_scale) = quantize(&self.policy_weights);
        let (value_weights, value_scale) = quantize(&self.value_weights);
        let mut packed_weights =
            Vec::with_capacity(input_weights.len() + policy_weights.len() + value_weights.len());
        packed_weights.extend(input_weights);
        packed_weights.extend(policy_weights);
        packed_weights.extend(value_weights);

        let mut output = String::new();
        writeln!(output, "pub const INPUT_SCALE: f32 = {input_scale:?};").unwrap();
        write_f32_array(&mut output, "INPUT_BIAS", &self.input_bias);
        writeln!(output, "\npub const POLICY_SCALE: f32 = {policy_scale:?};").unwrap();
        write_f32_array(&mut output, "POLICY_BIAS", &self.policy_bias);
        writeln!(output, "\npub const VALUE_SCALE: f32 = {value_scale:?};").unwrap();
        writeln!(output, "pub const VALUE_BIAS: f32 = {:?};", self.value_bias).unwrap();
        output.push_str("\npub const PACKED_WEIGHTS_HEX: &str = \"");
        for weight in packed_weights {
            write!(output, "{:02x}", weight as u8).unwrap();
        }
        output.push_str("\";\n");
        output
    }

    fn forward_hidden(&self, features: &[f32], hidden: &mut [f32]) {
        for (output, activation) in hidden.iter_mut().enumerate() {
            let offset = output * self.shape.input;
            let mut sum = self.input_bias[output];
            for (input, feature) in features.iter().enumerate() {
                sum += feature * self.input_weights[offset + input];
            }
            *activation = sum.max(0.0);
        }
    }

    fn forward_heads(&self, hidden: &[f32], policy: &mut [f32]) -> f32 {
        for (output, logit) in policy.iter_mut().enumerate() {
            let offset = output * self.shape.hidden;
            let mut sum = self.policy_bias[output];
            for (input, activation) in hidden.iter().enumerate() {
                sum += activation * self.policy_weights[offset + input];
            }
            *logit = sum;
        }
        self.value_weights
            .iter()
            .zip(hidden)
            .fold(self.value_bias, |sum, (weight, activation)| {
                sum + weight * activation
            })
            .tanh()
    }

    fn accumulate_gradient(
        &self,
        example: &TrainingExample<'_>,
        workspace: &mut TrainingWorkspace,
        gradient: &mut BatchGradient,
    ) {
        assert_eq!(example.features.len(), self.shape.input);
        assert_eq!(example.policy.len(), self.shape.policy);
        workspace.hidden.fill(0.0);
        for output in 0..self.shape.hidden {
            let offset = output * self.shape.input;
            let mut sum = self.input_bias[output];
            for input in 0..self.shape.input {
                sum += example.features[input] as f32 * self.input_weights[offset + input];
            }
            workspace.hidden[output] = sum.max(0.0);
        }

        for output in 0..self.shape.policy {
            let offset = output * self.shape.hidden;
            let mut sum = self.policy_bias[output];
            for input in 0..self.shape.hidden {
                sum += workspace.hidden[input] * self.policy_weights[offset + input];
            }
            workspace.logits[output] = sum;
        }
        let maximum = workspace
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut normalizer = 0.0;
        for action in 0..self.shape.policy {
            workspace.probabilities[action] = (workspace.logits[action] - maximum).exp();
            normalizer += workspace.probabilities[action];
        }
        for probability in &mut workspace.probabilities {
            *probability /= normalizer;
        }

        let value = self
            .value_weights
            .iter()
            .zip(&workspace.hidden)
            .fold(self.value_bias, |sum, (weight, activation)| {
                sum + weight * activation
            })
            .tanh();

        workspace.hidden_gradient.fill(0.0);
        for action in 0..self.shape.policy {
            let output_gradient = workspace.probabilities[action] - example.policy[action];
            let offset = action * self.shape.hidden;
            gradient.policy_bias[action] += output_gradient;
            for hidden_index in 0..self.shape.hidden {
                gradient.policy_weights[offset + hidden_index] +=
                    output_gradient * workspace.hidden[hidden_index];
                workspace.hidden_gradient[hidden_index] +=
                    output_gradient * self.policy_weights[offset + hidden_index];
            }
        }

        let value_gradient = 2.0 * (value - example.value) * (1.0 - value * value);
        gradient.value_bias += value_gradient;
        for hidden_index in 0..self.shape.hidden {
            gradient.value_weights[hidden_index] += value_gradient * workspace.hidden[hidden_index];
            workspace.hidden_gradient[hidden_index] +=
                value_gradient * self.value_weights[hidden_index];
            if workspace.hidden[hidden_index] == 0.0 {
                workspace.hidden_gradient[hidden_index] = 0.0;
            }
        }

        for hidden_index in 0..self.shape.hidden {
            let hidden_gradient = workspace.hidden_gradient[hidden_index];
            gradient.input_bias[hidden_index] += hidden_gradient;
            let offset = hidden_index * self.shape.input;
            for input in 0..self.shape.input {
                if example.features[input] != 0 {
                    gradient.input_weights[offset + input] +=
                        hidden_gradient * example.features[input] as f32;
                }
            }
        }

        let policy_loss = example
            .policy
            .iter()
            .zip(&workspace.probabilities)
            .filter(|(target, _)| **target > 0.0)
            .map(|(target, probability)| -target * probability.max(1e-12).ln())
            .sum::<f32>();
        gradient.loss += policy_loss + (value - example.value).powi(2);
        gradient.samples += 1;
    }

    fn batch_gradient(&self, examples: &[TrainingExample<'_>]) -> BatchGradient {
        assert!(!examples.is_empty(), "training batch is empty");
        examples
            .par_iter()
            .fold(
                || {
                    (
                        BatchGradient::new(self.shape),
                        TrainingWorkspace::new(self.shape),
                    )
                },
                |(mut gradient, mut workspace), example| {
                    self.accumulate_gradient(example, &mut workspace, &mut gradient);
                    (gradient, workspace)
                },
            )
            .map(|(gradient, _)| gradient)
            .reduce(|| BatchGradient::new(self.shape), BatchGradient::merge)
    }
}

fn add_slices(target: &mut [f32], source: &[f32]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += source;
    }
}

fn subtract_scaled(target: &mut [f32], gradient: &[f32], scale: f32) {
    for (target, gradient) in target.iter_mut().zip(gradient) {
        *target -= scale * gradient;
    }
}

fn adam_update_slice(
    parameters: &mut [f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    gradient: &[f32],
    update: AdamUpdate,
) {
    for (((parameter, first), second), gradient) in parameters
        .iter_mut()
        .zip(first_moment)
        .zip(second_moment)
        .zip(gradient)
    {
        adam_update_scalar(
            parameter,
            first,
            second,
            gradient * update.gradient_scale,
            update,
        );
    }
}

fn adam_update_scalar(
    parameter: &mut f32,
    first_moment: &mut f32,
    second_moment: &mut f32,
    gradient: f32,
    update: AdamUpdate,
) {
    *first_moment = 0.9 * *first_moment + 0.1 * gradient;
    *second_moment = 0.999 * *second_moment + 0.001 * gradient * gradient;
    let corrected_first = *first_moment / update.correction1;
    let corrected_second = *second_moment / update.correction2;
    *parameter -= update.learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8);
}

fn quantize(weights: &[f32]) -> (Vec<i8>, f32) {
    let maximum = weights.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
    let values = weights
        .iter()
        .map(|weight| (weight / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (values, scale)
}

fn write_f32_array(output: &mut String, name: &str, values: &[f32]) {
    write!(output, "pub const {name}: [f32; {}] = [", values.len()).unwrap();
    for value in values {
        write!(output, "{value:?},").unwrap();
    }
    output.push_str("];\n");
}

fn write_f32s(output: &mut impl Write, values: &[f32]) -> io::Result<()> {
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_parameter_set(output: &mut impl Write, parameters: &ParameterSet) -> io::Result<()> {
    write_f32s(output, &parameters.input_weights)?;
    write_f32s(output, &parameters.input_bias)?;
    write_f32s(output, &parameters.policy_weights)?;
    write_f32s(output, &parameters.policy_bias)?;
    write_f32s(output, &parameters.value_weights)?;
    output.write_all(&parameters.value_bias.to_le_bytes())
}

fn read_f32s(input: &mut impl Read, values: &mut [f32]) -> io::Result<()> {
    for value in values {
        let mut bytes = [0; 4];
        input.read_exact(&mut bytes)?;
        *value = f32::from_le_bytes(bytes);
    }
    Ok(())
}

fn read_parameter_set(input: &mut impl Read, parameters: &mut ParameterSet) -> io::Result<()> {
    read_f32s(input, &mut parameters.input_weights)?;
    read_f32s(input, &mut parameters.input_bias)?;
    read_f32s(input, &mut parameters.policy_weights)?;
    read_f32s(input, &mut parameters.policy_bias)?;
    read_f32s(input, &mut parameters.value_weights)?;
    let mut value_bias = [0; 4];
    input.read_exact(&mut value_bias)?;
    parameters.value_bias = f32::from_le_bytes(value_bias);
    Ok(())
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    fn symmetric(&mut self, magnitude: f32) -> f32 {
        let unit = self.next_u32() as f32 / u32::MAX as f32;
        (unit * 2.0 - 1.0) * magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("az-nn-{name}-{}.bin", std::process::id()))
    }

    #[test]
    fn checkpoint_rejects_wrong_shape() {
        let checkpoint = path("shape");
        let shape = MlpShape {
            input: 12,
            hidden: 5,
            policy: 3,
        };
        MlpNetwork::random(shape, 1).save(&checkpoint).unwrap();
        let wrong = MlpShape {
            input: 13,
            hidden: 5,
            policy: 3,
        };
        let error = match MlpNetwork::load(&checkpoint, wrong) {
            Ok(_) => panic!("wrong checkpoint shape was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        fs::remove_file(checkpoint).unwrap();
    }

    #[test]
    fn legacy_checkpoint_remains_loadable() {
        let checkpoint = path("legacy");
        let shape = MlpShape {
            input: 4,
            hidden: 3,
            policy: 2,
        };
        let network = MlpNetwork::random(shape, 2);
        let mut output = BufWriter::new(File::create(&checkpoint).unwrap());
        output.write_all(LEGACY_CHECKPOINT_MAGIC).unwrap();
        write_f32s(&mut output, &network.input_weights).unwrap();
        write_f32s(&mut output, &network.input_bias).unwrap();
        write_f32s(&mut output, &network.policy_weights).unwrap();
        write_f32s(&mut output, &network.policy_bias).unwrap();
        write_f32s(&mut output, &network.value_weights).unwrap();
        output.write_all(&network.value_bias.to_le_bytes()).unwrap();
        output.flush().unwrap();

        MlpNetwork::load(&checkpoint, shape).unwrap();
        fs::remove_file(checkpoint).unwrap();
    }

    #[test]
    fn one_item_batch_matches_single_sample_update() {
        let shape = MlpShape {
            input: 4,
            hidden: 3,
            policy: 2,
        };
        let mut single = MlpNetwork::random(shape, 3);
        let mut batch = single.clone();
        let features = [1, 0, 1, 0];
        let policy = [0.75, 0.25];
        let mut workspace = TrainingWorkspace::new(shape);
        single.train_sample(&features, &policy, 1.0, 0.01, &mut workspace);
        batch.train_batch(
            &[TrainingExample {
                features: &features,
                policy: &policy,
                value: 1.0,
            }],
            0.01,
        );

        assert_eq!(single.input_weights, batch.input_weights);
        assert_eq!(single.input_bias, batch.input_bias);
        assert_eq!(single.policy_weights, batch.policy_weights);
        assert_eq!(single.policy_bias, batch.policy_bias);
        assert_eq!(single.value_weights, batch.value_weights);
        assert_eq!(single.value_bias, batch.value_bias);
    }

    #[test]
    fn adam_resume_matches_uninterrupted_training() {
        let shape = MlpShape {
            input: 4,
            hidden: 3,
            policy: 2,
        };
        let initial = MlpNetwork::random(shape, 4);
        let features = [1, 0, 1, 0];
        let policy = [0.75, 0.25];
        let examples = [TrainingExample {
            features: &features,
            policy: &policy,
            value: 1.0,
        }];

        let mut uninterrupted = initial.clone();
        let mut uninterrupted_adam = AdamOptimizer::new(shape);
        uninterrupted.train_batch_adam(&examples, 0.001, &mut uninterrupted_adam);
        uninterrupted.train_batch_adam(&examples, 0.001, &mut uninterrupted_adam);

        let model_path = path("adam-model");
        let adam_path = path("adam-state");
        let mut resumed = initial;
        let mut resumed_adam = AdamOptimizer::new(shape);
        resumed.train_batch_adam(&examples, 0.001, &mut resumed_adam);
        resumed.save(&model_path).unwrap();
        resumed_adam.save(&adam_path).unwrap();
        let mut resumed = MlpNetwork::load(&model_path, shape).unwrap();
        let mut resumed_adam = AdamOptimizer::load(&adam_path, shape).unwrap();
        resumed.train_batch_adam(&examples, 0.001, &mut resumed_adam);

        assert_eq!(uninterrupted.input_weights, resumed.input_weights);
        assert_eq!(uninterrupted.input_bias, resumed.input_bias);
        assert_eq!(uninterrupted.policy_weights, resumed.policy_weights);
        assert_eq!(uninterrupted.policy_bias, resumed.policy_bias);
        assert_eq!(uninterrupted.value_weights, resumed.value_weights);
        assert_eq!(uninterrupted.value_bias, resumed.value_bias);
        assert_eq!(uninterrupted_adam.step(), resumed_adam.step());
        fs::remove_file(model_path).unwrap();
        fs::remove_file(adam_path).unwrap();
    }
}
