FINAL ANSWER

Edited only `core.b_in.weight`, using record-derived drug-response corrections with insulin gain 1.40 and norepinephrine gain 1.00. The final audit confirmed the submitted weights match those tested.

Expected readouts, based on matched tuning rollouts:

| Dose edit | Predicted mean change |
|---|---:|
| Insulin ×2 → glucose_finger | −14.90 |
| Insulin ×½ → glucose_finger | +5.94 |
| Norepinephrine ×2 → map_arterial | +5.61 |
| Norepinephrine ×½ → map_arterial | −2.59 |

Measured factual NLL increases were **+0.00375** on 400 tuning stays and **+0.00249** on 200 additional training stays, both below +0.005.

The main insulin response closely matches the record-based estimate; the other responses remain underestimated. Unseen-stay performance and the causal estimates remain uncertain.