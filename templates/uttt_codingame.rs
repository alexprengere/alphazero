use std::io::{self, BufRead};

use az_pipeline::uttt_deploy::DeploymentState;
use az_pipeline::uttt_network::Network;
use az_pipeline::uttt_schema::{action_from_row_col, action_to_row_col};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut state = DeploymentState::new();

    while let Some(Ok(line)) = lines.next() {
        let Some((opponent_row, opponent_col)) = parse_pair(&line) else {
            continue;
        };
        if opponent_row >= 0 {
            state.record_move(action_from_row_col(
                opponent_row as usize,
                opponent_col as usize,
            ));
        }

        let Some(Ok(count_line)) = lines.next() else {
            break;
        };
        let valid_count: usize = count_line.trim().parse().expect("valid action count");
        let mut valid_actions = Vec::with_capacity(valid_count);
        for _ in 0..valid_count {
            let line = lines.next().expect("valid action line").expect("stdin");
            let (row, col) = parse_pair(&line).expect("valid action");
            valid_actions.push(action_from_row_col(row as usize, col as usize));
        }

        let (logits, _) = Network::evaluate(&state.encode(&valid_actions));
        let action = valid_actions
            .into_iter()
            .max_by(|&left, &right| {
                let left_score = logits[left] + state.fallback_score(left);
                let right_score = logits[right] + state.fallback_score(right);
                left_score.total_cmp(&right_score)
            })
            .expect("at least one valid action");
        state.record_move(action);
        let (row, col) = action_to_row_col(action);
        println!("{row} {col}");
    }
}

fn parse_pair(line: &str) -> Option<(i32, i32)> {
    let mut values = line.split_whitespace().map(str::parse::<i32>);
    Some((values.next()?.ok()?, values.next()?.ok()?))
}
