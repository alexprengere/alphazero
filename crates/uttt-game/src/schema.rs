pub const BOARD_CELLS: usize = 81;
pub const MICRO_BOARDS: usize = 9;
pub const FEATURE_SIZE: usize = BOARD_CELLS * 4 + MICRO_BOARDS * 2;
pub const EMPTY: i8 = -1;
pub const DRAW: i8 = 2;

pub fn action_from_row_col(row: usize, col: usize) -> usize {
    row * 9 + col
}

pub fn action_to_row_col(action: usize) -> (usize, usize) {
    (action / 9, action % 9)
}

pub fn micro_board(action: usize) -> usize {
    let (row, col) = action_to_row_col(action);
    (row / 3) * 3 + col / 3
}

pub fn local_cell(action: usize) -> usize {
    let (row, col) = action_to_row_col(action);
    (row % 3) * 3 + col % 3
}
