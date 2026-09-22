# GENERATED — edit HUB.md, then: scripts/hubgen.py --write
- Depth lives in docs/ spokes: read the spoke before CHANGING what it covers.
- Launch from a node, never inside a src/. Only src/ recurses.
- Slots exist only when nonempty. Ephemera (build/, scratch/, worktrees) are
  gitignored; archive/ is frozen.
- Put a thing at the LCA of its consumers. Everything enters at the bottom.
- Filesystem should resemble the shape of the project, Conway style:
  give real crates, programs, and subsystems their own folders.
- Internal structure changes are clean migrations: update callers and
  remove retired paths; do not retain compatibility shims.
- DER00: Leave the code you touched a tiny bit better. The requested change
  can supply the improvement; no extra cleanup is owed. Preserve required
  behavior except intended changes. Do not expand scope, invent abstractions,
  or move complexity elsewhere merely to satisfy this rule.
- Read and improve the owning scope's docs/ as you learn. Roughly every 5-25
  turns, fold useful findings, ideas, corrections, or next steps into local
  docs; a small edit is enough. Do not rely on chat or compaction alone.
- Child hubs only ADD — never restate or override an ancestor.
- Ratchet baselines only go down; an exception needs a written waiver.
- Verify by driving the real thing; report red gates verbatim.
- Commit frequently: small working increments, in the child repo that owns
  the change. Never end a task with the work only on disk.
- Shell cwd persists across tool calls and resets between them unpredictably:
  use absolute paths; never trust a bare `cd`.
- Doctrine: <root>/docs/ · why hubs look like this: <root>/docs/hubs.md
# --- end invariant — this node's half follows ---
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
