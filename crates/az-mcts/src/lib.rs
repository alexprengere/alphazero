use std::sync::Arc;

use az_core::{Evaluator, Game};

#[derive(Clone)]
struct Edge<A> {
    action: A,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

#[derive(Default)]
struct Node {
    visits: u32,
    value_sum: f32,
    expanded: bool,
    edge_start: usize,
    edge_len: usize,
}

pub struct SearchResult<A> {
    pub action: A,
    pub visits: Vec<(A, u32)>,
    pub reused_root_visits: u32,
}

#[derive(Clone, Copy)]
pub struct RootNoise {
    pub alpha: f32,
    pub epsilon: f32,
    pub seed: u64,
}

pub struct Mcts<G: Game> {
    nodes: Vec<Node>,
    edges: Vec<Edge<G::Action>>,
    legal_actions: Vec<G::Action>,
    policy_logits: Vec<f32>,
    root: usize,
    c_puct: f32,
    evaluator: Arc<dyn Evaluator<G>>,
}

impl<G: Game> Mcts<G> {
    pub fn new(c_puct: f32, evaluator: Arc<dyn Evaluator<G>>) -> Self {
        Self {
            nodes: Vec::with_capacity(4096),
            edges: Vec::with_capacity(32768),
            legal_actions: Vec::with_capacity(128),
            policy_logits: Vec::new(),
            root: 0,
            c_puct,
            evaluator,
        }
    }

    pub fn search(&mut self, root: &G, simulations: usize) -> SearchResult<G::Action> {
        self.search_with_noise(root, simulations, None)
    }

    pub fn search_with_noise(
        &mut self,
        root: &G,
        simulations: usize,
        noise: Option<RootNoise>,
    ) -> SearchResult<G::Action> {
        if self.nodes.is_empty() {
            self.nodes.push(Node::default());
            self.root = 0;
        }
        let root_index = self.root;
        let reused_root_visits = self.nodes[root_index].visits;
        let mut state = root.clone();
        if !self.nodes[root_index].expanded {
            self.expand(root_index, &state);
        }
        let noisy_root = noise.is_some();
        if let Some(noise) = noise {
            self.apply_root_noise(root_index, noise);
        }

        let additional_simulations = if noisy_root {
            simulations
        } else {
            simulations.saturating_sub(reused_root_visits as usize)
        };
        for _ in 0..additional_simulations {
            state.clone_from(root);
            self.simulate(root_index, &mut state);
        }

        let root_edges = self.edge_range(root_index);
        let visits = self.edges[root_edges.clone()]
            .iter()
            .map(|edge| (edge.action, edge.visits))
            .collect();
        let action = self.edges[root_edges]
            .iter()
            .max_by_key(|edge| edge.visits)
            .map(|edge| edge.action)
            .expect("search called on terminal state");
        SearchResult {
            action,
            visits,
            reused_root_visits,
        }
    }

    pub fn advance(&mut self, action: G::Action) {
        let child = self.nodes.get(self.root).and_then(|_| {
            self.edges[self.edge_range(self.root)]
                .iter()
                .find(|edge| edge.action == action)
                .and_then(|edge| edge.child)
        });
        if let Some(child) = child {
            self.root = child;
        } else {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.root = 0;
    }

    fn simulate(&mut self, node_index: usize, state: &mut G) -> f32 {
        if state.is_terminal() {
            return state.outcome(state.current_player());
        }
        if !self.nodes[node_index].expanded {
            let value = self.expand(node_index, state);
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += value;
            return value;
        }

        let edge_index = self.select_edge(node_index);
        let action = self.edges[edge_index].action;
        state.play(action);

        let child_index = match self.edges[edge_index].child {
            Some(index) => index,
            None => {
                let index = self.nodes.len();
                self.nodes.push(Node::default());
                self.edges[edge_index].child = Some(index);
                index
            }
        };

        let value = -self.simulate(child_index, state);
        let edge = &mut self.edges[edge_index];
        edge.visits += 1;
        edge.value_sum += value;
        let node = &mut self.nodes[node_index];
        node.visits += 1;
        node.value_sum += value;
        value
    }

    fn select_edge(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        let parent_scale = (node.visits.max(1) as f32).sqrt();
        let edge_start = node.edge_start;
        self.edges[self.edge_range(node_index)]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                let left_score = edge_score(left, parent_scale, self.c_puct);
                let right_score = edge_score(right, parent_scale, self.c_puct);
                left_score.total_cmp(&right_score)
            })
            .map(|(index, _)| edge_start + index)
            .expect("expanded node has no legal actions")
    }

    fn expand(&mut self, node_index: usize, state: &G) -> f32 {
        self.policy_logits.resize(state.action_space(), 0.0);
        self.policy_logits.fill(0.0);
        let value = self.evaluator.evaluate(state, &mut self.policy_logits);

        self.legal_actions.clear();
        state.legal_actions(&mut self.legal_actions);
        let max_logit = self
            .legal_actions
            .iter()
            .map(|&action| self.policy_logits[state.action_index(action)])
            .fold(f32::NEG_INFINITY, f32::max);
        let normalizer: f32 = self
            .legal_actions
            .iter()
            .map(|&action| (self.policy_logits[state.action_index(action)] - max_logit).exp())
            .sum();

        let edge_start = self.edges.len();
        self.edges.reserve(self.legal_actions.len());
        for &action in &self.legal_actions {
            self.edges.push(Edge {
                action,
                prior: (self.policy_logits[state.action_index(action)] - max_logit).exp()
                    / normalizer.max(1e-12),
                visits: 0,
                value_sum: 0.0,
                child: None,
            });
        }
        self.nodes[node_index] = Node {
            visits: 0,
            value_sum: 0.0,
            expanded: true,
            edge_start,
            edge_len: self.legal_actions.len(),
        };
        value
    }

    fn edge_range(&self, node_index: usize) -> std::ops::Range<usize> {
        let node = &self.nodes[node_index];
        node.edge_start..node.edge_start + node.edge_len
    }

    fn apply_root_noise(&mut self, node_index: usize, noise: RootNoise) {
        assert!(noise.alpha > 0.0, "Dirichlet alpha must be positive");
        assert!(
            (0.0..=1.0).contains(&noise.epsilon),
            "Dirichlet epsilon must be between zero and one"
        );
        let range = self.edge_range(node_index);
        let count = range.len();
        if count == 0 || noise.epsilon == 0.0 {
            return;
        }
        let mut rng = Rng::new(noise.seed);
        let mut samples = Vec::with_capacity(count);
        let mut total = 0.0;
        for _ in 0..count {
            let sample = sample_gamma(noise.alpha, &mut rng);
            samples.push(sample);
            total += sample;
        }
        for (edge, sample) in self.edges[range].iter_mut().zip(samples) {
            let random_prior = sample / total;
            edge.prior = (1.0 - noise.epsilon) * edge.prior + noise.epsilon * random_prior;
        }
    }
}

fn edge_score<A>(edge: &Edge<A>, parent_scale: f32, c_puct: f32) -> f32 {
    let q = if edge.visits == 0 {
        0.0
    } else {
        edge.value_sum / edge.visits as f32
    };
    q + c_puct * edge.prior * parent_scale / (1 + edge.visits) as f32
}

fn sample_gamma(alpha: f32, rng: &mut Rng) -> f32 {
    if alpha < 1.0 {
        return sample_gamma(alpha + 1.0, rng) * rng.uniform().powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let normal = rng.normal();
        let base = 1.0 + c * normal;
        if base <= 0.0 {
            continue;
        }
        let value = base * base * base;
        let uniform = rng.uniform();
        if uniform < 1.0 - 0.0331 * normal.powi(4)
            || uniform.ln() < 0.5 * normal * normal + d * (1.0 - value + value.ln())
        {
            return d * value;
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    fn uniform(&mut self) -> f32 {
        ((self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0)) as f32
    }

    fn normal(&mut self) -> f32 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_samples_are_positive_and_deterministic() {
        let mut left = Rng::new(42);
        let mut right = Rng::new(42);
        for _ in 0..100 {
            let left_sample = sample_gamma(0.3, &mut left);
            let right_sample = sample_gamma(0.3, &mut right);
            assert!(left_sample > 0.0);
            assert_eq!(left_sample, right_sample);
        }
    }
}
