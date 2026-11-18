# Reusable Rust AlphaZero Prototype

This is a first end-to-end version of the deployment architecture described in
`ALPHAZERO_PLAN.md`:

- separate Ultimate Tic-Tac-Toe and Connect Four game crates;
- PUCT MCTS teacher for offline self-play;
- Rust policy/value learner;
- int8 Rust weight export;
- policy-network Codingame bot with a lightweight tactical fallback and no MCTS
  at runtime.

Training checkpoints and generated submissions are local artifacts. Train a
checkpoint before evaluating or packaging a learned policy.

## Reusable architecture

The reusable pipeline is split from the game implementation:

```text
crates/
├── az-core/     # Game and policy/value evaluator contracts
├── az-mcts/     # Generic PUCT search and tree reuse
├── az-nn/       # Trainable MLP, checkpointing, inference, and quantization
├── connect4-game/ # Connect Four rules, features, and replay
└── uttt-game/     # Ultimate Tic-Tac-Toe rules, features, replay, and deployment
src/
├── uttt_adapter.rs      # UTTT evaluator/search-result bridge
├── connect4_adapter.rs  # Connect Four evaluator/model bridge
└── bin/         # Self-play, training, loop, viewer, play, and packaging tools
templates/
├── uttt_network.rs      # Codingame quantized-inference template
└── uttt_codingame.rs    # Codingame protocol template
```

To add another two-player zero-sum game:

1. create a game crate implementing `az_core::Game`;
2. define its action indexing and feature encoder;
3. implement an `az_core::Evaluator` adapter for its heuristic or network;
4. define its replay records and model shape; and
5. reuse `az_mcts::Mcts` and `az_nn::MlpNetwork`.

## Connect Four extensibility example

Connect Four lives independently in `crates/connect4-game/` and uses the same
generic PUCT and policy/value MLP crates.

Run an iterative Connect Four learning loop:

```bash
cargo run --release --bin alpha-loop -- \
  --game connect4 \
  --iterations 10 \
  --games 500 \
  --simulations 400 \
  --threads 8 \
  --dirichlet-alpha 0.3 \
  --dirichlet-epsilon 0.25 \
  --epochs 5 \
  --optimizer adam \
  --batch-size 256 \
  --replay-window 5 \
  --learning-rate 0.001
```

The loop and export tools use game-specific default paths:

```text
data/connect4/iterations/
models/connect4-current.bin
models/connect4-current.adam
models/connect4-generated-weights.rs  # created by export-model
```

UTTT replay shards are stored separately in:

```text
data/uttt/iterations/
```

Play against the current checkpoint:

```bash
cargo run --release --bin connect4-play
```

Play as the second player:

```bash
cargo run --release --bin connect4-play -- --human O
```

## Generate self-play data

```bash
cargo run --release --bin uttt-selfplay -- \
  --games 100 --simulations 400 --threads 8 \
  --output data/uttt/selfplay.bin
```

The first bootstrap uses heuristic policy/value estimates inside MCTS. After
training, load the full-precision checkpoint for later self-play:

```bash
cargo run --release --bin uttt-selfplay -- \
  --games 100 --simulations 400 \
  --checkpoint models/uttt-current.bin \
  --output data/uttt/selfplay-v2.bin
```

## Watch a self-play game

```bash
cargo run --release --bin uttt-watch-selfplay -- \
  --simulations 200 --delay-ms 300
```

After training, use the full-precision checkpoint for policy/value evaluation:

```bash
cargo run --release --bin uttt-watch-selfplay -- \
  --checkpoint models/uttt-current.bin \
  --simulations 200 --delay-ms 300
```

The latest move is shown in lowercase. Set `--delay-ms 0` to print every frame
without animation, and use `--seed N` to reproduce a game.

## Play UTTT against the latest checkpoint

Play as `X`:

```bash
cargo run --release --bin uttt-play
```

Play as `O`, letting the policy move first:

```bash
cargo run --release --bin uttt-play -- --human O
```

Enter moves as `row column`, with both coordinates from `0` to `8`. Enter `q`
to stop. Add `--no-clear` if the terminal does not handle ANSI screen clearing.
The tool loads the latest full-precision checkpoint from
`models/uttt-current.bin`. Use another checkpoint with `--checkpoint PATH`.

## Train and export

```bash
cargo run --release --bin uttt-train -- \
  data/uttt/selfplay.bin \
  --epochs 20 \
  --optimizer adam \
  --batch-size 256 \
  --learning-rate 0.001 \
  --checkpoint models/uttt-current.bin
cargo test
```

The trainer saves the continuing full-precision checkpoint to
`models/uttt-current.bin`. Adam momentum and variance are saved beside it in
`models/uttt-current.adam`. Use `--optimizer sgd` for plain mini-batch SGD.

Export an existing checkpoint as quantized Rust source:

```bash
cargo run --release --bin export-model -- --game uttt
```

For Connect Four:

```bash
cargo run --release --bin export-model -- --game connect4
```

## Run the iterative AlphaZero loop

```bash
cargo run --release --bin alpha-loop -- \
  --game uttt \
  --iterations 10 \
  --games 1000 \
  --simulations 400 \
  --threads 8 \
  --dirichlet-alpha 0.3 \
  --dirichlet-epsilon 0.25 \
  --epochs 5 \
  --optimizer adam \
  --batch-size 256 \
  --replay-window 5 \
  --arena-games 100 \
  --arena-simulations 200 \
  --promotion-threshold 0.55 \
  --learning-rate 0.001
```

The first iteration bootstraps from heuristic MCTS and promotes its initial
network automatically. Each later iteration:

1. loads the current champion for self-play;
2. generates a new self-play shard;
3. copies the champion model and Adam state into a candidate;
4. trains the candidate on the recent replay window;
5. runs paired, color-swapped arena matches from randomized openings; and
6. promotes the candidate only when its score reaches the threshold.

Rejected candidates are deleted, leaving the champion checkpoint unchanged.
The arena score is `(wins + 0.5 × draws) / games`, with a default `0.55`
promotion threshold.

Self-play mixes Dirichlet noise into legal root priors:

```text
noisy_prior = 0.75 × network_prior + 0.25 × Dirichlet(0.3)
```

Arena matches and human-vs-policy play remain noise-free. The
`uttt-watch-selfplay` tool uses the same root-noise defaults as self-play. Set
`--dirichlet-epsilon 0` to disable root noise.

`--threads` controls parallel self-play, mini-batch gradient calculation, and
arena matches. If omitted, the available CPU parallelism is used. Dataset
records are still written in deterministic game order. MCTS reuses the selected
child subtree between moves. Noise-free searches count inherited visits toward
the `--simulations` budget; noisy self-play performs the requested number of
new simulations after reusing the subtree.

`alpha-loop` uses the iteration number as the self-play seed, so an unchanged
champion still explores a different deterministic stream in each iteration.
Standalone self-play defaults to seed `0`; pass `--seed N` to select another
reproducible stream.

Rerunning the loop appends after the highest existing iteration shard and
reuses the current float checkpoint. It does not transactionally resume a
partially completed iteration, so stop it between iterations when possible.
The Rust binaries are compiled once before the loop rather than once per
iteration. Afterward, generate the submission:

```bash
cargo run --release --bin uttt-package
```

## Run the Codingame bot

```bash
rustc --edition=2021 -O submission/uttt/main.rs -o submission/uttt/bot
```

The generated bot accepts the standard Ultimate Tic-Tac-Toe protocol: opponent
row and column, valid-action count, then that many valid row/column pairs. It
selects only from the referee-provided actions.

## Generate the single-file submission

```bash
cargo run --release --bin uttt-package
rustc -O submission/uttt/main.rs -o submission/uttt/bot
```

The packager includes only the lightweight deployment state, feature encoding,
network inference, tactical fallback, packed weights, and Codingame protocol
loop. The full game simulator and MCTS teacher are not included. It reads
`models/uttt-current.bin` directly; no generated Rust weight file is required.

## Current limitations

- MCTS performs scalar rather than batched neural inference.
- Neural-network training and inference run on the CPU; there is no GPU backend.
- Loop recovery is iteration-based rather than transactional within an
  iteration.
- Single-file Codingame packaging is implemented for UTTT, not Connect Four.
- Exported int8 weights use two-character hex packing. Base85 or a custom
  high-density encoding could reduce source size further.
