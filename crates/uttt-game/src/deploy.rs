use crate::schema::{
    local_cell, micro_board, BOARD_CELLS, DRAW, EMPTY, FEATURE_SIZE, MICRO_BOARDS,
};

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

pub struct DeploymentState {
    cells: [i8; BOARD_CELLS],
    micro: [i8; MICRO_BOARDS],
    current_player: i8,
}

impl Default for DeploymentState {
    fn default() -> Self {
        Self::new()
    }
}

impl DeploymentState {
    pub fn new() -> Self {
        Self {
            cells: [EMPTY; BOARD_CELLS],
            micro: [EMPTY; MICRO_BOARDS],
            current_player: 0,
        }
    }

    pub fn record_move(&mut self, action: usize) {
        let player = self.current_player;
        self.cells[action] = player;
        let board = micro_board(action);
        if self.board_winner(board, player) {
            self.micro[board] = player;
        } else if self
            .board_actions(board)
            .all(|candidate| self.cells[candidate] != EMPTY)
        {
            self.micro[board] = DRAW;
        }
        self.current_player = 1 - player;
    }

    pub fn encode(&self, legal: &[usize]) -> [f32; FEATURE_SIZE] {
        let mut features = [0.0; FEATURE_SIZE];
        let player = self.current_player;
        let opponent = 1 - player;

        for (index, &owner) in self.cells.iter().enumerate() {
            if owner == player {
                features[index] = 1.0;
            } else if owner == opponent {
                features[BOARD_CELLS + index] = 1.0;
            }
        }
        for &action in legal {
            features[BOARD_CELLS * 2 + action] = 1.0;
        }

        if let Some(&first) = legal.first() {
            let board = micro_board(first);
            if legal.iter().all(|&action| micro_board(action) == board) {
                for action in self.board_actions(board) {
                    features[BOARD_CELLS * 3 + action] = 1.0;
                }
            }
        }

        for (board, &owner) in self.micro.iter().enumerate() {
            if owner == player {
                features[BOARD_CELLS * 4 + board] = 1.0;
            } else if owner == opponent {
                features[BOARD_CELLS * 4 + MICRO_BOARDS + board] = 1.0;
            }
        }
        features
    }

    pub fn fallback_score(&self, action: usize) -> f32 {
        let player = self.current_player;
        let opponent = 1 - player;
        let board = micro_board(action);
        let local = local_cell(action);
        let mut score = match local {
            4 => 0.30,
            0 | 2 | 6 | 8 => 0.15,
            _ => 0.05,
        };
        if self.would_win_board(board, action, player) {
            score += 8.0;
        }
        if self.would_win_board(board, action, opponent) {
            score += 3.0;
        }
        score
    }

    fn board_actions(&self, board: usize) -> impl Iterator<Item = usize> {
        (0..9).map(move |local| board_action(board, local))
    }

    fn board_winner(&self, board: usize, player: i8) -> bool {
        LINES.iter().any(|line| {
            line.iter()
                .all(|&local| self.cells[board_action(board, local)] == player)
        })
    }

    fn would_win_board(&self, board: usize, action: usize, player: i8) -> bool {
        LINES.iter().any(|line| {
            line.iter().all(|&local| {
                let candidate = board_action(board, local);
                candidate == action || self.cells[candidate] == player
            })
        })
    }
}

fn board_action(board: usize, local: usize) -> usize {
    let row = (board / 3) * 3 + local / 3;
    let col = (board % 3) * 3 + local % 3;
    row * 9 + col
}
