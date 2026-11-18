use crate::game::{Game, CELLS, COLUMNS};

pub const FEATURE_SIZE: usize = CELLS * 2 + COLUMNS;

pub fn encode(game: &Game) -> [f32; FEATURE_SIZE] {
    let mut features = [0.0; FEATURE_SIZE];
    let player = game.current_player();
    let opponent = 1 - player;
    for (cell, &owner) in game.cells().iter().enumerate() {
        if owner == player {
            features[cell] = 1.0;
        } else if owner == opponent {
            features[CELLS + cell] = 1.0;
        }
    }
    for column in 0..COLUMNS {
        if game.is_legal(column) {
            features[CELLS * 2 + column] = 1.0;
        }
    }
    features
}
