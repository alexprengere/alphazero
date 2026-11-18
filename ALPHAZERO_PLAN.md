# AlphaZero-Style Bot Plan

## Goal

Build a Rust-first system that:

1. reproduces the Codingame multiplayer game accurately and quickly;
2. generates training data through self-play;
3. trains a policy/value neural network from MCTS targets;
4. evaluates new models against baselines and previous champions; and
5. exports a small, dependency-free Rust bot that fits Codingame's source-size,
   memory, compiler, and per-turn time limits.

The training stack may use Python and a standard ML framework, but the simulator,
search implementation, model format, inference code, and final bot should share
Rust-compatible definitions. Deployment constraints are design inputs, not a
final cleanup phase.

## Success criteria

- The simulator passes deterministic rule tests and differential tests against
  recorded referee games.
- Self-play can run reproducibly across many processes from a model checkpoint.
- MCTS produces legal actions and improves over raw network play.
- Candidate models are promoted only after statistically meaningful matches.
- The final generated Rust submission uses only APIs supported by Codingame.
- The deployed model and inference implementation stay within explicit source
  size, memory, startup, and turn-time budgets.
- The deployed bot is measurably stronger than heuristic, search-only, and
  network-only baselines.

Record the actual platform limits as soon as the contest opens:

| Constraint | Initial target | Confirmed limit |
| --- | ---: | ---: |
| First-turn time | 500 ms | TBD |
| Later-turn time | 40 ms | TBD |
| Submission source | 100 KB | TBD |
| Runtime memory | 64 MB | TBD |
| Rust compiler | Pin after probing | TBD |

Initial targets should be stricter than the platform limits so local success
survives Codingame overhead.

## Proposed repository layout

```text
2027/
├── Cargo.toml
├── crates/
│   ├── az-core/          # Generic game and evaluator contracts
│   ├── az-mcts/          # Reusable PUCT and tree storage
│   ├── az-nn/            # Training, checkpoint, inference, and quantization
│   └── uttt-game/        # Example game rules, features, and replay format
├── src/
│   ├── bin/              # Pipeline and Codingame command-line applications
│   ├── mcts.rs           # Game-specific evaluator adapters
│   └── network.rs        # Embedded deployment-network adapter
├── models/               # Local artifacts; checkpoints ignored by Git
└── submission/           # Generated final source and metadata
```

Keep generic search and neural-network crates independent of any game rules.
The final `bot` should be buildable with the same Rust version available on
Codingame.

## Core interfaces

Define these contracts before optimizing implementation details:

```rust
trait GameState: Clone {
    type Action: Copy + Eq;

    fn current_player(&self) -> Player;
    fn legal_actions(&self, out: &mut ActionBuffer);
    fn apply(&mut self, action: Self::Action) -> Transition;
    fn undo(&mut self, transition: Transition);
    fn is_terminal(&self) -> bool;
    fn outcome(&self, player: Player) -> f32;
    fn observation(&self, player: Player, out: &mut FeatureBuffer);
    fn action_index(&self, action: Self::Action) -> usize;
}
```

The concrete API may differ, but it must support:

- allocation-free legal-action generation in hot paths;
- fast apply/undo or cheap state cloning;
- stable action indices shared by Rust and Python;
- player-relative observations and values;
- deterministic state hashing;
- terminal outcomes, draws, and score-based tie breakers;
- simultaneous moves, chance events, or hidden information if the game uses
  them.

For turns containing several commands, avoid an uncontrolled combinatorial
macro-action space. Prefer an autoregressive state machine:

1. generate one atomic command;
2. update a temporary within-turn state;
3. repeat until an explicit `END_TURN` action.

If command ordering is irrelevant, canonicalize it to prevent duplicate action
sequences. If the game is simultaneous, represent a complete turn as a joint
transition and define exactly what information each player may observe.

## Phase 1: Rules and simulator

### 1. Capture the game specification

- Translate every rule into an executable checklist.
- Document turn order, legal actions, resource changes, combat resolution,
  tie-breaking, terminal conditions, and input/output formats.
- Identify hidden information, randomness, and referee ambiguities early.
- Save representative Codingame input streams and match replays as fixtures.

### 2. Implement a correctness-first simulator

- Start with clear data structures and immutable identifiers.
- Separate static map data from mutable state.
- Use compact integer IDs and contiguous arrays where practical.
- Make randomness explicit through a seeded RNG passed into transitions.
- Return errors for impossible transitions in tests; do not silently repair
  invalid states.

### 3. Test simulator correctness

- Unit-test each rule and boundary condition.
- Add property tests for invariants such as resource conservation, occupancy,
  unique IDs, legal masks, and terminal-state stability.
- Test `apply` followed by `undo` against byte-for-byte or hash equality.
- Replay captured games and compare each state with referee observations.
- When possible, run differential matches between the local simulator and the
  official referee using identical actions.

### 4. Optimize only after conformance

Benchmark:

- state clone versus apply/undo;
- legal-action generation;
- feature extraction;
- terminal checks;
- complete random games;
- simulations per millisecond.

Then replace maps and heap allocations with fixed-capacity vectors, bitsets,
small arrays, or packed representations where profiling justifies it.

**Exit gate:** deterministic random-vs-random games run without invariant
failures, replay fixtures match, and simulator throughput supports the intended
MCTS budget.

## Phase 2: Baselines and evaluation harness

Implement progressively stronger opponents:

1. random legal agent;
2. simple rule-based agent;
3. shallow heuristic search or rollout agent;
4. policy-only neural agent;
5. MCTS with uniform priors and heuristic/rollout values;
6. neural MCTS agent.

The arena must support:

- swapped player positions and map seeds;
- deterministic match manifests;
- parallel matches;
- win/draw/loss and game-score reporting;
- Elo or Glicko estimates with confidence intervals;
- paired comparisons for candidate promotion;
- replay retention for failures and surprising games.

Maintain a small opponent league containing heuristics, recent checkpoints,
historical champions, and deliberately unusual strategies. Training only
against the latest model risks cycles and catastrophic forgetting.

**Exit gate:** arena results are reproducible and detect a deliberately stronger
baseline.

## Phase 3: State, action, and network design

### Observation encoding

Use player-relative, normalized features. Candidate inputs include:

- spatial planes for terrain, ownership, units, cooldowns, and objectives;
- scalar features for resources, scores, turn number, and remaining budgets;
- masks or embeddings for entity types;
- a short history if the observation is not Markov without it.

Keep the feature schema versioned and define it once in a machine-readable form
used to generate both Rust and Python constants. Add golden tests comparing
Rust features with the learner's decoded features.

### Policy encoding

- Assign every representable atomic action a stable global index.
- Mask illegal actions before softmax and renormalize legal probabilities.
- Assert that each generated legal action maps to exactly one policy index.
- If the action space is too large, factor the policy into heads such as action
  type, source, target, and parameter, while preserving exact legality.

### Initial network

Start deliberately small:

- a compact MLP for small non-spatial states; or
- a shallow residual CNN for fixed grids;
- shared trunk;
- policy head producing action logits;
- value head producing expected outcome in `[-1, 1]`.

Avoid operators that are expensive to reproduce in tiny Rust inference. Favor
linear layers, small convolutions, ReLU or clipped ReLU, additions, pooling, and
simple normalization folded into weights at export time.

The training model should have a clear path to an even smaller student model.
Do not make the first competitive network the minimum deployable network.

## Phase 4: MCTS

Implement PUCT search:

```text
score(s, a) = Q(s, a)
            + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
```

Each simulation:

1. selects actions using PUCT and legal masks;
2. applies transitions until an unexpanded or terminal state;
3. evaluates the leaf with the policy/value network;
4. expands legal children;
5. backs up the value with the correct player perspective;
6. undoes transitions or discards the cloned state.

Required features:

- root Dirichlet noise during self-play only;
- temperature-based action selection early in games;
- deterministic argmax or low temperature for evaluation;
- tree reuse between real turns;
- transposition support only if benchmarks show a benefit;
- a hard wall-clock guard for Codingame turns;
- an optional fixed-simulation mode for reproducible tests;
- batched leaf inference for offline self-play;
- virtual loss if several workers share one tree.

For multiplayer games with more than two players or non-zero-sum scoring,
replace scalar negation with a value vector or a clearly justified scalar
utility. For hidden-information games, standard AlphaZero MCTS is insufficient;
use information-set observations and evaluate determinization, re-determinizing
ISMCTS, or a recurrent policy.

**Exit gate:** uniform-prior MCTS beats the random agent, neural MCTS is legal
and deterministic under a fixed seed, and search fits a measured time budget.

## Phase 5: Self-play data pipeline

Use an actor/learner architecture:

```text
checkpoint -> self-play actors -> replay shards -> learner
                                      |
                                      +-> arena and diagnostics
```

For every decision, store:

- encoded observation;
- legal-action mask or compact legal-action list;
- MCTS visit distribution;
- selected action;
- acting player;
- final outcome/value target;
- game, rules, feature-schema, and model versions.

Prefer a compact append-only binary format with:

- fixed endianness;
- checksums;
- optional per-shard compression;
- an index for random sample access;
- atomic write-then-rename publication.

The learner should consume completed immutable shards. This makes training
"offline" with respect to a frozen replay snapshot, even though new snapshots
are continually generated by self-play.

Self-play controls:

- seeded and reproducible games;
- randomized starting positions/maps where supported;
- opening diversity;
- resignation disabled initially, then enabled only after calibration;
- configurable MCTS simulations, exploration noise, and temperature;
- rejection and logging of corrupt samples;
- metrics for game length, outcome balance, policy entropy, value error, and
  unique-state rate.

Use a sliding replay window rather than only the newest checkpoint. Sample
recent data heavily while retaining enough historical data to reduce forgetting.

## Phase 6: Training loop

Train with the AlphaZero objective:

```text
loss = policy_cross_entropy(mcts_visits, policy_logits)
     + value_weight * value_loss(final_outcome, predicted_value)
     + regularization
```

Training requirements:

- mixed precision where safe;
- train/validation splits by complete game, not individual position;
- deterministic checkpoint metadata;
- optimizer and replay-window state saved for resumption;
- gradient, policy entropy, value calibration, and overfitting metrics;
- configurable symmetry augmentation for truly equivalent board transforms;
- class/replay balancing if one outcome dominates;
- periodic export and Rust inference conformance tests.

Iteration:

1. self-play with the current champion;
2. append data to the replay window;
3. train a candidate for a bounded number of updates;
4. export the candidate;
5. run candidate-versus-champion and league matches;
6. promote only if the confidence threshold is met;
7. repeat.

Bootstrap with heuristic or uniform MCTS targets if pure self-play fails to
learn. Expert trajectories may seed the buffer, but self-play targets should
eventually dominate.

## Phase 7: Model evaluation and promotion

Do not promote on training loss alone. A candidate must pass:

- Rust/Python output equivalence within a declared tolerance;
- inference and MCTS latency budgets;
- model/source-size budgets;
- no-illegal-action stress tests;
- paired arena matches against the current champion;
- league matches against historical and heuristic agents;
- representative-map and starting-player balance checks.

Use sequential probability ratio testing or a fixed match count with a
predeclared confidence interval. Keep promotion settings separate from tuning
matches to avoid repeatedly overfitting the gate.

## Phase 8: Compression and Rust inference

Compression is a staged optimization:

### 1. Establish the deployable inference subset

Implement a tiny Rust inference engine supporting only the required operators.
Use flat arrays and statically known tensor shapes. Avoid allocation after
startup.

### 2. Export a canonical float model

- Fold normalization and biases into adjacent layers.
- Reorder weights into the exact layout consumed by Rust.
- Export a small versioned binary for local testing.
- Compare every layer against Python on golden inputs.

### 3. Reduce model size

Evaluate, in this order:

1. architecture reduction using latency/strength sweeps;
2. knowledge distillation from a larger teacher;
3. structured channel or neuron pruning;
4. int8 quantization with per-channel weight scales;
5. quantization-aware training;
6. lower-bit weights only if int8 is insufficient;
7. entropy coding only if decompression cost and source encoding are worthwhile.

Prefer structured reductions because they reduce both source size and runtime.
Unstructured sparsity is useful only when the tiny inference engine can exploit
it without large index overhead.

### 4. Embed weights in the final source

Compare:

- decimal arrays: simplest but source-heavy;
- hexadecimal byte arrays: predictable and easy to decode;
- Base64/Ascii85 text: smaller textual representation with startup decoding;
- custom packed integer literals: potentially smallest but harder to maintain.

Generate the representation; never hand-edit weights. The packager should emit
the model hash, schema version, architecture constants, and compile-time shape
checks.

### 5. Optimize inference

- Use integer accumulators with calibrated scaling.
- Fuse bias, activation, and requantization.
- Specialize loops for fixed dimensions.
- Reuse fixed scratch buffers.
- Benchmark scalar loops before adding unsafe or architecture-specific code.
- Keep a portable path because Codingame CPU features may be unknown.

### 6. Optimize the search/model tradeoff

Measure playing strength under the real turn budget across:

- larger model with fewer simulations;
- smaller model with more simulations;
- policy-only fallback on timeout;
- reduced branching through policy top-k while preserving mandatory tactical
  actions.

The best model in isolation may not produce the best complete bot.

**Exit gate:** the generated submission compiles with the Codingame Rust
version, matches exported-model outputs, stays inside all budgets, and completes
stress-test turns before the local safety deadline.

## Phase 9: Codingame packaging

Generate a single `main.rs` containing:

- input parser;
- compact game state;
- legal-action generator and transition logic needed by search;
- MCTS;
- compressed weights;
- fixed-shape inference;
- action formatter;
- stderr diagnostics behind a compile-time constant.

The packaging tool should:

1. inline selected modules in dependency order;
2. strip tests, training-only APIs, comments if necessary, and debug logging;
3. embed generated weights;
4. compile with the pinned Codingame-compatible toolchain;
5. report source bytes, binary size, startup time, inference latency, and MCTS
   throughput;
6. run fixture and arena smoke tests;
7. write a manifest containing Git commit, model hash, and configuration.

Keep the readable multi-crate source authoritative.
`submission/uttt/main.rs` is a generated artifact.

## Performance and reliability safeguards

- Reserve a safety margin and stop MCTS before the platform deadline.
- Use a monotonic clock and check it at bounded intervals.
- Always retain the best legal action found so far.
- Fall back to the network policy or a deterministic heuristic if no simulation
  completes.
- Never emit an illegal or malformed command on timeout.
- Avoid startup work that can be generated ahead of time.
- Log compact timing and search statistics to stderr during development.
- Fuzz the parser and legal-action/output round trip.

## Experiment tracking

Every run should record:

- Git commit and dirty-state flag;
- rules, feature, action, and replay schema versions;
- model architecture and parameter count;
- optimizer and learning-rate schedule;
- replay shard IDs and sampling weights;
- self-play and MCTS parameters;
- random seeds;
- arena opponents and results;
- exported source size and Rust latency.

A lightweight JSON manifest plus CSV/SQLite metrics is sufficient initially.
Do not make a hosted experiment service a prerequisite for iteration.

## Milestones

### M0: Project skeleton

- Rust workspace and Python environment.
- Shared schema/version constants.
- CI commands for formatting, linting, unit tests, and packaging smoke tests.

### M1: Correct simulator

- Parser, state, actions, transitions, terminal scoring.
- Fixtures, property tests, replay tool, and random-game stress tests.

### M2: Search-ready engine

- Allocation-conscious simulation API.
- Random and heuristic agents.
- Arena and benchmarks.
- Uniform/heuristic MCTS.

### M3: First learning loop

- Observation/action encoders.
- Small policy/value model.
- Self-play shards and replay loader.
- Training, export, and Rust/Python conformance.

### M4: Iterative AlphaZero pipeline

- Parallel actors.
- Replay window.
- Candidate/champion promotion.
- Opponent league and experiment manifests.

### M5: Competitive bot

- Tuned MCTS and network.
- Distillation and quantization-aware training.
- Time-control testing under Codingame-like conditions.

### M6: Submission compression

- Tiny fixed-shape inference.
- Generated embedded weights.
- Single-file Rust packager.
- Source-size, latency, and legality gates.

## Recommended implementation order

1. Create the Rust workspace and simulator test harness.
2. Implement rules plus a random legal agent.
3. Build the arena before any neural-network work.
4. Add heuristic evaluation and MCTS to validate the search API.
5. Freeze version 1 of observation and action encodings.
6. Train a tiny policy/value model on self-play data.
7. Implement float inference in Rust and exact export tests.
8. Close the actor/learner/promotion loop.
9. Tune playing strength with a larger teacher model.
10. Distill and quantize a deployment student.
11. Generate and benchmark the single-file submission continuously.

## Early decisions to revisit when the game is known

- Perfect versus hidden information.
- Deterministic versus stochastic transitions.
- Alternating, simultaneous, or multi-command turns.
- Fixed versus variable map and entity counts.
- Two-player zero-sum versus multiplayer/general-sum values.
- Maximum practical global action space.
- Whether board symmetries preserve all rules.
- Actual Codingame compiler, source, memory, and time limits.

These choices can change the correct search algorithm and network outputs, so
resolve them before locking the replay schema or producing large datasets.
