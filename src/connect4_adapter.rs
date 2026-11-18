use std::sync::Arc;

use az_core::Evaluator;
use az_nn::{MlpNetwork, MlpShape};
use connect4_game::features::{encode, FEATURE_SIZE};
use connect4_game::game::{Game, COLUMNS};

pub const HIDDEN_SIZE: usize = 64;

pub fn model_shape() -> MlpShape {
    MlpShape {
        input: FEATURE_SIZE,
        hidden: HIDDEN_SIZE,
        policy: COLUMNS,
    }
}

pub fn heuristic_evaluator() -> Arc<dyn Evaluator<Game>> {
    Arc::new(Connect4Evaluator::Heuristic)
}

pub fn checkpoint_evaluator(network: Arc<MlpNetwork>) -> Arc<dyn Evaluator<Game>> {
    Arc::new(Connect4Evaluator::Checkpoint(network))
}

enum Connect4Evaluator {
    Heuristic,
    Checkpoint(Arc<MlpNetwork>),
}

impl Evaluator<Game> for Connect4Evaluator {
    fn evaluate(&self, state: &Game, policy_logits: &mut [f32]) -> f32 {
        match self {
            Self::Heuristic => {
                for (column, logit) in policy_logits.iter_mut().enumerate() {
                    *logit = state.tactical_score(column);
                }
                state.heuristic_value(state.current_player())
            }
            Self::Checkpoint(network) => network.evaluate(&encode(state), policy_logits),
        }
    }
}
