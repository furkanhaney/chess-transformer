# Migration contract

Source: `../research/src/learning/chessformer/` at the Rasat source roof. The selected
experiment is `runs/run_009_value`, because it exercises the 5M Chessformer,
the policy and value objectives, history, GAB, AdamW, and the largest recorded
local corpus (100,000 games; 783,251 delivered positions).

## Preserved in the Axis acceptance path

- one token per board square and twelve piece/color channels per position;
- concatenated current and historical positions;
- vertical square mirroring plus color swap for Black-to-move examples;
- moves transformed into the normalized board coordinates;
- pre-LayerNorm residual transformer blocks with bidirectional attention;
- the small-model, mean-pooled geometric attention bias path;
- per-layer GAB mixing networks with one shared square-pair template matrix;
- two-times-width GELU feed-forward expansion and no dropout;
- scaled bilinear source/destination logits over all 64 × 64 traversals;
- mean-pooled win/draw/loss prediction from the moving side's perspective;
- `policy_loss + 0.1 * value_loss`, AdamW, blank-free categorical targets; and
- finite repeated passes with stable IDs and whole-game train/eval separation.

## Deliberately different or deferred

- The acceptance model is tiny and f32. It does not reproduce the 5.08M
  parameter scale, mixed precision, throughput, or accuracy of the Python run.
- Eight checked opening lines replace the Lichess archive for the bounded
  smoke. They establish mechanics only, not generalization or playing strength.
- Promotion-piece logits are deferred. All acceptance moves are ordinary
  source/destination moves, so the policy objective remains complete for this
  corpus but does not test the paper's additive promotion bias.
- Elo conditioning is deferred. The Python run enabled it in configuration,
  while its own report says the embeddings were loaded but not used.
- Axis uses its current GELU implementation and Xavier-style `Linear`
  initialization rather than PyTorch's explicit normal initialization.
- Checkpointing, AMP, Hydra, plotting, and the NumPy memory-map format are not
  carried over. They are experiment plumbing, not compatibility surfaces.

## Historical claim boundary

`create_efficient_dataloaders()` in the Python source constructs one dataset
with `split=None`, then chooses validation indices from that same dataset while
continuing to train over all of it. The reported 59.01% policy and 52.88% value
"validation" accuracies from run 009 therefore include positions seen during
training. They remain historical training-subset diagnostics.

The Axis smoke splits game lines first and uses the complete normalized UCI
move sequence as the semantic game identity. Axis retains evaluation game
identities and checks every streamed training position against them. This
proves observed whole-game separation for the included corpus, including
duplicate move sequences that happen to carry different source-row IDs. It
does not prove the corpus is representative of Lichess or ALLIE.

## Full-run admission

A future accuracy claim needs all of the following in one durable run record:

1. source archive hashes and the exact time-control/time-pressure filters;
2. Elo-bin counts before and after balancing;
3. game IDs assigned to one split before position sampling;
4. model/configuration, Axis revision, seed, hardware, and elapsed time;
5. finite, completed optimizer state and sample/pass counts; and
6. policy, value, promotion, and legal-probability metrics on an independent
   game corpus, with ALLIE named only when that corpus is actually used.
