# Axis acceptance witness

Date: 2026-09-20

This is the first bounded end-to-end witness for the Rust migration. It is a
mechanics result over the checked local opening corpus, not a chess-strength or
generalization result.

## Environment

- GPU: NVIDIA GeForce RTX 5060, 8,151 MiB
- driver: 595.84
- Rust: 1.96.1
- Axis: `767326049ef661e226a074f3fe551ebcd98613bf`
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
initial total_loss=8.689576 policy_loss=8.578979 value_loss=1.105971
final   total_loss=7.676713 policy_loss=7.566498 value_loss=1.102147

population:             4
declared passes:        1
completed passes:       1
total observations:     4

unique training IDs:    4
unique evaluation IDs:  2
observed overlap:        0
```

Policy accuracy remained 0.00% and value accuracy remained 50.00% on the two
evaluation positions. The useful evidence is finite forward/backward/update,
a lower held-out objective in this deterministic smoke, an exact completed
pass, and zero observed identity overlap. The sample is too small for the
accuracy values or loss movement to support a model-quality claim.
