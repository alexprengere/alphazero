pub trait Game: Clone {
    type Action: Copy + Eq + Send + Sync + 'static;

    fn current_player(&self) -> i8;
    fn is_terminal(&self) -> bool;
    fn outcome(&self, player: i8) -> f32;
    fn play(&mut self, action: Self::Action);
    fn legal_actions(&self, out: &mut Vec<Self::Action>);
    fn action_index(&self, action: Self::Action) -> usize;
    fn action_space(&self) -> usize;
}

pub trait Evaluator<G: Game>: Send + Sync {
    fn evaluate(&self, state: &G, policy_logits: &mut [f32]) -> f32;
}
