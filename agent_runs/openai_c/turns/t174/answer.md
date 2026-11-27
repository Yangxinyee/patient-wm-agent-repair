FINAL ANSWER

Kept two edits to strengthen weak dose sensitivity:
- `embed.numeric.weight`: rank-one gain **4** along the joint dose/rate input direction.
- `core.b_in.weight`: scaled by **1.30**.

Expected readouts, based on pooled rollouts:

| Dose edit | Expected shift | Record-estimated effect |
|---|---:|---:|
| Insulin ×2 → glucose | −2.81 | −12.51 |
| Insulin ×½ → glucose | +2.01 | +6.26 |
| Norepinephrine ×2 → MAP | +0.13 | +4.67 |
| Norepinephrine ×½ → MAP | −0.18 | −2.34 |

Factual NLL increased **0.00315 nats** across 300 stays; its one-sided 95% upper bound was **0.00477**, below the budget.

This is **only a partial repair**: substantial underestimation remains, and norepinephrine halving did not improve over the original.