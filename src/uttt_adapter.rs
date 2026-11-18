use std::sync::Arc;

use az_core::Evaluator;
use az_nn::{MlpNetwork, MlpShape};

use crate::uttt_features::encode;
use crate::uttt_game::{Game, BOARD_CELLS};
use crate::uttt_schema::FEATURE_SIZE;

pub use az_mcts::RootNoise;

pub const HIDDEN_SIZE: usize = 64;

pub fn model_shape() -> MlpShape {
    MlpShape {
        input: FEATURE_SIZE,
        hidden: HIDDEN_SIZE,
        policy: BOARD_CELLS,
    }
}

#[derive(Clone)]
pub enum EvaluatorKind {
    Heuristic,
    Checkpoint(Arc<MlpNetwork>),
}

impl Evaluator<Game> for EvaluatorKind {
    fn evaluate(&self, state: &Game, policy_logits: &mut [f32]) -> f32 {
        match self {
            Self::Heuristic => {
                for (action, logit) in policy_logits.iter_mut().enumerate() {
                    *logit = state.tactical_score(action);
                }
                state.heuristic_value(state.current_player())
            }
            Self::Checkpoint(network) => network.evaluate(&encode(state), policy_logits),
        }
    }
}

pub struct SearchResult {
    pub action: usize,
    pub visits: [u32; BOARD_CELLS],
    pub reused_root_visits: u32,
}

pub struct Mcts {
    inner: az_mcts::Mcts<Game>,
}

impl Mcts {
    pub fn new(c_puct: f32, evaluator: EvaluatorKind) -> Self {
        Self {
            inner: az_mcts::Mcts::new(c_puct, Arc::new(evaluator)),
        }
    }

    pub fn search(&mut self, root: &Game, simulations: usize) -> SearchResult {
        let result = self.inner.search(root, simulations);
        Self::convert_result(result)
    }

    pub fn search_with_noise(
        &mut self,
        root: &Game,
        simulations: usize,
        noise: RootNoise,
    ) -> SearchResult {
        let result = self.inner.search_with_noise(root, simulations, Some(noise));
        Self::convert_result(result)
    }

    fn convert_result(result: az_mcts::SearchResult<usize>) -> SearchResult {
        let mut visits = [0; BOARD_CELLS];
        for (action, count) in result.visits {
            visits[action] = count;
        }
        SearchResult {
            action: result.action,
            visits,
            reused_root_visits: result.reused_root_visits,
        }
    }

    pub fn advance(&mut self, action: usize) {
        self.inner.advance(action);
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_a_legal_action_and_visits() {
        let game = Game::new();
        let mut search = Mcts::new(1.5, EvaluatorKind::Heuristic);
        let result = search.search(&game, 32);
        assert!(game.is_legal(result.action));
        assert!(result.visits.iter().sum::<u32>() > 0);
    }

    #[test]
    fn selected_child_visits_are_reused() {
        let mut game = Game::new();
        let mut search = Mcts::new(1.5, EvaluatorKind::Heuristic);
        let first = search.search(&game, 64);
        game.play(first.action);
        search.advance(first.action);

        let second = search.search(&game, 1);
        assert!(second.reused_root_visits > 0);
        assert!(game.is_legal(second.action));
    }
}
