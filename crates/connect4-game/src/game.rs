pub const COLUMNS: usize = 7;
pub const ROWS: usize = 6;
pub const CELLS: usize = COLUMNS * ROWS;
pub const EMPTY: i8 = -1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    cells: [i8; CELLS],
    heights: [u8; COLUMNS],
    current_player: i8,
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
            cells: [EMPTY; CELLS],
            heights: [0; COLUMNS],
            current_player: 0,
            winner: EMPTY,
            ply: 0,
        }
    }

    pub fn cells(&self) -> &[i8; CELLS] {
        &self.cells
    }

    pub fn current_player(&self) -> i8 {
        self.current_player
    }

    pub fn is_terminal(&self) -> bool {
        self.winner != EMPTY || self.ply as usize == CELLS
    }

    pub fn winner(&self) -> Option<i8> {
        match self.winner {
            0 | 1 => Some(self.winner),
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

    pub fn is_legal(&self, column: usize) -> bool {
        column < COLUMNS && !self.is_terminal() && self.heights[column] < ROWS as u8
    }

    pub fn legal_actions(&self, out: &mut Vec<usize>) {
        out.clear();
        if self.is_terminal() {
            return;
        }
        for (column, &height) in self.heights.iter().enumerate() {
            if height < ROWS as u8 {
                out.push(column);
            }
        }
    }

    pub fn play(&mut self, column: usize) {
        assert!(self.is_legal(column), "illegal column {column}");
        let row = self.heights[column] as usize;
        self.cells[index(row, column)] = self.current_player;
        self.heights[column] += 1;
        self.ply += 1;
        if self.has_four(row, column, self.current_player) {
            self.winner = self.current_player;
        }
        self.current_player = 1 - self.current_player;
    }

    pub fn tactical_score(&self, column: usize) -> f32 {
        if !self.is_legal(column) {
            return f32::NEG_INFINITY;
        }
        let player = self.current_player;
        let opponent = 1 - player;
        let mut score = 0.4 - (column as f32 - 3.0).abs() * 0.08;
        if self.would_win(column, player) {
            score += 100.0;
        }
        if self.would_win(column, opponent) {
            score += 8.0;
        }
        score
    }

    pub fn heuristic_value(&self, player: i8) -> f32 {
        if self.is_terminal() {
            return self.outcome(player);
        }
        let opponent = 1 - player;
        let center: f32 = (0..ROWS)
            .map(|row| self.cells[index(row, COLUMNS / 2)])
            .fold(0.0, |score, owner| {
                score
                    + if owner == player {
                        0.08
                    } else if owner == opponent {
                        -0.08
                    } else {
                        0.0
                    }
            });
        center.clamp(-0.8, 0.8)
    }

    fn would_win(&self, column: usize, player: i8) -> bool {
        if self.heights[column] >= ROWS as u8 {
            return false;
        }
        let row = self.heights[column] as usize;
        self.count_line(row, column, 1, 0, player) + self.count_line(row, column, -1, 0, player) + 1
            >= 4
            || self.count_line(row, column, 0, 1, player)
                + self.count_line(row, column, 0, -1, player)
                + 1
                >= 4
            || self.count_line(row, column, 1, 1, player)
                + self.count_line(row, column, -1, -1, player)
                + 1
                >= 4
            || self.count_line(row, column, 1, -1, player)
                + self.count_line(row, column, -1, 1, player)
                + 1
                >= 4
    }

    fn has_four(&self, row: usize, column: usize, player: i8) -> bool {
        self.count_line(row, column, 1, 0, player) + self.count_line(row, column, -1, 0, player) + 1
            >= 4
            || self.count_line(row, column, 0, 1, player)
                + self.count_line(row, column, 0, -1, player)
                + 1
                >= 4
            || self.count_line(row, column, 1, 1, player)
                + self.count_line(row, column, -1, -1, player)
                + 1
                >= 4
            || self.count_line(row, column, 1, -1, player)
                + self.count_line(row, column, -1, 1, player)
                + 1
                >= 4
    }

    fn count_line(
        &self,
        row: usize,
        column: usize,
        delta_column: isize,
        delta_row: isize,
        player: i8,
    ) -> usize {
        let mut count = 0;
        let mut column = column as isize + delta_column;
        let mut row = row as isize + delta_row;
        while column >= 0 && column < COLUMNS as isize && row >= 0 && row < ROWS as isize {
            if self.cells[index(row as usize, column as usize)] != player {
                break;
            }
            count += 1;
            column += delta_column;
            row += delta_row;
        }
        count
    }
}

pub fn index(row: usize, column: usize) -> usize {
    row * COLUMNS + column
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_fall_to_lowest_empty_row() {
        let mut game = Game::new();
        game.play(3);
        game.play(3);
        assert_eq!(game.cells[index(0, 3)], 0);
        assert_eq!(game.cells[index(1, 3)], 1);
    }

    #[test]
    fn detects_horizontal_win() {
        let mut game = Game::new();
        for column in [0, 0, 1, 1, 2, 2, 3] {
            game.play(column);
        }
        assert_eq!(game.winner(), Some(0));
    }

    #[test]
    fn detects_vertical_win() {
        let mut game = Game::new();
        for column in [3, 4, 3, 4, 3, 4, 3] {
            game.play(column);
        }
        assert_eq!(game.winner(), Some(0));
    }

    #[test]
    fn detects_diagonal_win() {
        let mut game = Game::new();
        for column in [0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3, 3] {
            game.play(column);
        }
        assert_eq!(game.winner(), Some(0));
    }

    #[test]
    fn full_column_is_illegal() {
        let mut game = Game::new();
        for _ in 0..ROWS {
            game.play(0);
        }
        assert!(!game.is_legal(0));
    }
}
