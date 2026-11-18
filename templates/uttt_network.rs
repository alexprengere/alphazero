use std::sync::OnceLock;

use crate::uttt_schema::{BOARD_CELLS, FEATURE_SIZE};

mod weights {
    include!("uttt_generated_weights.rs");
}

pub const HIDDEN_SIZE: usize = 64;
const INPUT_WEIGHT_COUNT: usize = FEATURE_SIZE * HIDDEN_SIZE;
const POLICY_WEIGHT_COUNT: usize = HIDDEN_SIZE * BOARD_CELLS;
const TOTAL_WEIGHT_COUNT: usize = INPUT_WEIGHT_COUNT + POLICY_WEIGHT_COUNT + HIDDEN_SIZE;

static PACKED_WEIGHTS: OnceLock<Vec<i8>> = OnceLock::new();

pub struct Network;

impl Network {
    pub fn evaluate(features: &[f32; FEATURE_SIZE]) -> ([f32; BOARD_CELLS], f32) {
        let packed = PACKED_WEIGHTS.get_or_init(decode_weights);
        let input_weights = &packed[..INPUT_WEIGHT_COUNT];
        let policy_weights = &packed[INPUT_WEIGHT_COUNT..INPUT_WEIGHT_COUNT + POLICY_WEIGHT_COUNT];
        let value_weights = &packed[INPUT_WEIGHT_COUNT + POLICY_WEIGHT_COUNT..];

        let mut hidden = [0.0; HIDDEN_SIZE];
        for (output, activation) in hidden.iter_mut().enumerate() {
            let offset = output * FEATURE_SIZE;
            let mut sum = weights::INPUT_BIAS[output];
            for (input, feature) in features.iter().enumerate() {
                sum += feature * input_weights[offset + input] as f32 * weights::INPUT_SCALE;
            }
            *activation = sum.max(0.0);
        }

        let mut policy = [0.0; BOARD_CELLS];
        for (output, logit) in policy.iter_mut().enumerate() {
            let offset = output * HIDDEN_SIZE;
            let mut sum = weights::POLICY_BIAS[output];
            for (input, activation) in hidden.iter().enumerate() {
                sum += activation * policy_weights[offset + input] as f32 * weights::POLICY_SCALE;
            }
            *logit = sum;
        }

        let mut value = weights::VALUE_BIAS;
        for (input, activation) in hidden.iter().enumerate() {
            value += activation * value_weights[input] as f32 * weights::VALUE_SCALE;
        }

        fn decode_weights() -> Vec<i8> {
            if weights::PACKED_WEIGHTS_HEX.is_empty() {
                return vec![0; TOTAL_WEIGHT_COUNT];
            }
            assert_eq!(weights::PACKED_WEIGHTS_HEX.len(), TOTAL_WEIGHT_COUNT * 2);
            (0..TOTAL_WEIGHT_COUNT)
                .map(|index| {
                    let offset = index * 2;
                    let bytes = weights::PACKED_WEIGHTS_HEX.as_bytes();
                    ((hex_digit(bytes[offset]) << 4) | hex_digit(bytes[offset + 1])) as i8
                })
                .collect()
        }

        fn hex_digit(value: u8) -> u8 {
            match value {
                b'0'..=b'9' => value - b'0',
                b'a'..=b'f' => value - b'a' + 10,
                _ => panic!("invalid packed weight"),
            }
        }
        (policy, value.tanh())
    }
}
