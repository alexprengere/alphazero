use crate::game::{micro_board, Game};
use crate::schema::{BOARD_CELLS, FEATURE_SIZE, MICRO_BOARDS};

pub fn encode(game: &Game) -> [f32; FEATURE_SIZE] {
    let mut features = [0.0; FEATURE_SIZE];
    let player = game.current_player();
    let opponent = 1 - player;

    for (index, &owner) in game.cells().iter().enumerate() {
        if owner == player {
            features[index] = 1.0;
        } else if owner == opponent {
            features[BOARD_CELLS + index] = 1.0;
        }
    }

    let mut legal = [0; BOARD_CELLS];
    let legal_count = game.legal_actions_into(&mut legal);
    for &action in &legal[..legal_count] {
        features[BOARD_CELLS * 2 + action] = 1.0;
    }

    if let Some(board) = game.forced_board() {
        for action in 0..BOARD_CELLS {
            if micro_board(action) == board {
                features[BOARD_CELLS * 3 + action] = 1.0;
            }
        }
    }

    for (board, &owner) in game.micro_boards().iter().enumerate() {
        if owner == player {
            features[BOARD_CELLS * 4 + board] = 1.0;
        } else if owner == opponent {
            features[BOARD_CELLS * 4 + MICRO_BOARDS + board] = 1.0;
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_features_include_legal_mask() {
        let features = encode(&Game::new());
        assert_eq!(
            features[BOARD_CELLS * 2..BOARD_CELLS * 3]
                .iter()
                .sum::<f32>(),
            81.0
        );
    }
}
