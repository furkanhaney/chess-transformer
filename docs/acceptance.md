# Axis acceptance witness

Date: 2026-09-20

This is the first bounded end-to-end witness for the Rust migration. It is a
mechanics result over the checked local opening corpus, not a chess-strength or
generalization result.

## Environment

- GPU: NVIDIA GeForce RTX 5060, 8,151 MiB
- driver: 595.84
- Rust: 1.96.1
- Axis: crates.io `0.3.0` (registry checksum
  `baa22e690d5e9cf76ee91183935698a71a69c3be8093c2d07f2f40f62a60b238`)
- corpus SHA-256:
  `a5ecaa4e1464e945db0196a4d71a8f3c7678bd6dcb943d38bc6e06da7abdf6cb`

Command, from the repository root, reusing the already provisioned sibling
CUDA 13.2 toolkit:

```bash
CUDA_TOOLKIT_PATH="$(realpath ../sudoku-transformer/.cuda)" bash scripts/check.sh
```

## Result

Formatting, Clippy with warnings denied, and two representation/data tests
passed. The smoke built a 12,331-parameter model with one GAB transformer
layer, embedding width 12, three heads, and two bias templates. It trained for
one complete shuffled pass: four positions in two AdamW updates. Two positions
from a held-out game were evaluated before and after training.

```text
initial total_loss=8.689576 policy_loss=8.578979 value_loss=1.105970
final   total_loss=7.676713 policy_loss=7.566498 value_loss=1.102147

FINITE PASSES STATUS

population:             4
declared passes:        1
completed passes:       1
total observations:     4

PASS
SEMANTIC DISJOINTNESS

identity scheme:       chess-move-sequence@1
evaluation: mode=retained observations=2 unique=1 repeats=1
training: mode=streaming observations=4 unique=not-retained repeats=not-measured
observed cross-population overlap: 0

PASS
```

Policy accuracy remained 0.00% and value accuracy remained 50.00% on the two
evaluation positions. The useful evidence is finite forward/backward/update,
a lower held-out objective in this deterministic smoke, an exact completed
pass, and zero observed overlap under exact UCI move-sequence identity. The
retained evaluation population contains two positions from one game, hence one
unique game identity and one within-population repeat. The sample is too small
for the accuracy values or loss movement to support a model-quality claim.
