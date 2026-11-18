pub use crate::schema::{
    action_from_row_col, action_to_row_col, local_cell, micro_board, BOARD_CELLS, DRAW, EMPTY,
    MICRO_BOARDS,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    cells: [i8; BOARD_CELLS],
    micro: [i8; MICRO_BOARDS],
    current_player: i8,
    forced_board: i8,
    winner: i8,
    ply: u8,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        Self {
            cells: [EMPTY; BOARD_CELLS],
            micro: [EMPTY; MICRO_BOARDS],
            current_player: 0,
            forced_board: EMPTY,
            winner: EMPTY,
            ply: 0,
        }
    }

    pub fn current_player(&self) -> i8 {
        self.current_player
    }

    pub fn forced_board(&self) -> Option<usize> {
        (self.forced_board >= 0).then_some(self.forced_board as usize)
    }

    pub fn cells(&self) -> &[i8; BOARD_CELLS] {
        &self.cells
    }

    pub fn micro_boards(&self) -> &[i8; MICRO_BOARDS] {
        &self.micro
    }

    pub fn is_terminal(&self) -> bool {
        self.winner != EMPTY || self.micro.iter().all(|&board| board != EMPTY)
    }

    pub fn winner(&self) -> Option<i8> {
        match self.winner {
            0 | 1 => Some(self.winner),
            _ if self.is_terminal() => {
                let p0 = self.micro.iter().filter(|&&owner| owner == 0).count();
                let p1 = self.micro.iter().filter(|&&owner| owner == 1).count();
                match p0.cmp(&p1) {
                    std::cmp::Ordering::Greater => Some(0),
                    std::cmp::Ordering::Less => Some(1),
                    std::cmp::Ordering::Equal => None,
                }
            }
            _ => None,
        }
    }

    pub fn outcome(&self, player: i8) -> f32 {
        match self.winner() {
            Some(winner) if winner == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    pub fn is_legal(&self, action: usize) -> bool {
        if action >= BOARD_CELLS || self.is_terminal() || self.cells[action] != EMPTY {
            return false;
        }
        let board = micro_board(action);
        if self.micro[board] != EMPTY {
            return false;
        }
        self.forced_board < 0 || board == self.forced_board as usize
    }

    pub fn legal_actions(&self, out: &mut Vec<usize>) {
        out.clear();
        let mut actions = [0; BOARD_CELLS];
        let count = self.legal_actions_into(&mut actions);
        out.extend_from_slice(&actions[..count]);
    }

    pub fn legal_actions_into(&self, out: &mut [usize; BOARD_CELLS]) -> usize {
        let mut count = 0;
        if self.is_terminal() {
            return count;
        }
        if let Some(board) = self.forced_board() {
            for action in board_actions(board) {
                if self.cells[action] == EMPTY {
                    out[count] = action;
                    count += 1;
                }
            }
            return count;
        }
        for action in 0..BOARD_CELLS {
            if self.cells[action] == EMPTY && self.micro[micro_board(action)] == EMPTY {
                out[count] = action;
                count += 1;
            }
        }
        count
    }

    pub fn play(&mut self, action: usize) {
        assert!(self.is_legal(action), "illegal action {action}");
        let player = self.current_player;
        self.cells[action] = player;
        self.ply += 1;

        let board = micro_board(action);
        if board_winner(&self.cells, board, player) {
            self.micro[board] = player;
        } else if board_actions(board).all(|index| self.cells[index] != EMPTY) {
            self.micro[board] = DRAW;
        }

        if line_winner(&self.micro, player) {
            self.winner = player;
        }

        let next_board = local_cell(action);
        self.forced_board = if self.micro[next_board] == EMPTY {
            next_board as i8
        } else {
            EMPTY
        };
        self.current_player = 1 - player;
    }

    pub fn tactical_score(&self, action: usize) -> f32 {
        if !self.is_legal(action) {
            return f32::NEG_INFINITY;
        }
        let player = self.current_player;
        let opponent = 1 - player;
        let board = micro_board(action);
        let local = local_cell(action);
        let mut score = match local {
            4 => 0.30,
            0 | 2 | 6 | 8 => 0.15,
            _ => 0.05,
        };

        let mut next = self.clone();
        next.play(action);
        if next.winner == player {
            score += 100.0;
        } else if next.micro[board] == player && self.micro[board] == EMPTY {
            score += 8.0;
        }

        if would_win_board(&self.cells, board, action, opponent) {
            score += 3.0;
        }
        if self.micro[local] != EMPTY {
            score -= 0.4;
        }
        score
    }

    pub fn heuristic_value(&self, player: i8) -> f32 {
        if self.is_terminal() {
            return self.outcome(player);
        }
        let opponent = 1 - player;
        let won = self.micro.iter().filter(|&&owner| owner == player).count() as f32;
        let lost = self
            .micro
            .iter()
            .filter(|&&owner| owner == opponent)
            .count() as f32;
        ((won - lost) / 5.0).clamp(-0.9, 0.9)
    }
}

fn board_actions(board: usize) -> impl Iterator<Item = usize> {
    (0..9).map(move |local| board_action(board, local))
}

fn board_action(board: usize, local: usize) -> usize {
    let row = (board / 3) * 3 + local / 3;
    let col = (board % 3) * 3 + local % 3;
    action_from_row_col(row, col)
}

fn board_winner(cells: &[i8; BOARD_CELLS], board: usize, player: i8) -> bool {
    LINES.iter().any(|line| {
        line.iter()
            .all(|&local| cells[board_action(board, local)] == player)
    })
}

fn would_win_board(cells: &[i8; BOARD_CELLS], board: usize, action: usize, player: i8) -> bool {
    LINES.iter().any(|line| {
        line.iter().all(|&local| {
            let candidate = board_action(board, local);
            candidate == action || cells[candidate] == player
        })
    })
}

fn line_winner(board: &[i8; MICRO_BOARDS], player: i8) -> bool {
    LINES
        .iter()
        .any(|line| line.iter().all(|&index| board[index] == player))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_move_allows_every_cell() {
        let game = Game::new();
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        assert_eq!(legal.len(), 81);
    }

    #[test]
    fn move_forces_matching_local_board() {
        let mut game = Game::new();
        game.play(action_from_row_col(1, 2));
        assert_eq!(game.forced_board(), Some(5));

        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        assert_eq!(legal.len(), 9);
        assert!(legal.iter().all(|&action| micro_board(action) == 5));
    }

    #[test]
    fn completed_target_board_releases_next_player() {
        let mut game = Game::new();
        for action in [
            action_from_row_col(0, 0),
            action_from_row_col(1, 0),
            action_from_row_col(0, 1),
            action_from_row_col(1, 1),
            action_from_row_col(0, 2),
        ] {
            game.cells[action] = 0;
        }
        game.micro[0] = 0;
        game.forced_board = 1;
        game.current_player = 1;
        game.play(action_from_row_col(0, 3));
        assert_eq!(game.forced_board(), None);
    }

    #[test]
    fn detects_local_board_win() {
        let mut game = Game::new();
        game.cells[action_from_row_col(0, 0)] = 0;
        game.cells[action_from_row_col(0, 1)] = 0;
        game.forced_board = 0;
        game.play(action_from_row_col(0, 2));
        assert_eq!(game.micro[0], 0);
    }
}
