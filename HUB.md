# Chess Transformer — human-move modelling on Axis

This is the Rust + Axis successor to `research/src/learning/chessformer`, the 2026 Python +
PyTorch replication of Chessformer. README.md is for outside humans. Migration
contracts, claim boundaries, and continuation state live under `docs/`.

The scientific object is human move prediction, not engine strength. Preserve
the 64-square tokens, side-to-move normalization, geometric attention bias,
source-to-destination policy, and side-to-move outcome target. Split whole games
before deriving positions. Never describe positions from training games as an
independent validation set.

Run `bash scripts/check.sh`. The local opening corpus is an architecture smoke,
not a playing-strength benchmark. A full experiment needs a versioned Lichess
recipe, time-control filtering, Elo balancing, a game-disjoint test corpus, and
a durable run certificate.

Axis is an outside crates.io dependency pinned in `Cargo.lock`. Shared tensor,
optimizer, data-regime, or module behavior belongs in Axis. Chess
representation, GAB, policy/value metrics, data recipes, and experiments belong
here. Upgrade Axis through a reviewed dependency PR and run the complete GPU
acceptance gate before merging it.
