pub mod data;
pub mod deploy;
pub mod features;
pub mod game;
pub mod schema;

use az_core::Game as GameContract;
use game::Game;

impl GameContract for Game {
    type Action = usize;

    fn current_player(&self) -> i8 {
        self.current_player()
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal()
    }

    fn outcome(&self, player: i8) -> f32 {
        self.outcome(player)
    }

    fn play(&mut self, action: Self::Action) {
        self.play(action);
    }

    fn legal_actions(&self, out: &mut Vec<Self::Action>) {
        self.legal_actions(out);
    }

    fn action_index(&self, action: Self::Action) -> usize {
        action
    }

    fn action_space(&self) -> usize {
        schema::BOARD_CELLS
    }
}
