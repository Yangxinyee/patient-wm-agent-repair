FINAL ANSWER

The final audit confirmed exact checkpoint integrity. I retained edits to the four value-bin head tensors and the value-bin transition embedding, encoding treatment corrections estimated from training records with compensating offsets to preserve factual fit.

Expected readouts, based on all 400 tuning stays:

| Dose edit | Predicted mean shift |
|---|---:|
| Insulin ×2 → glucose_finger | −12.84 |
| Insulin ×½ → glucose_finger | +5.78 |
| Norepinephrine ×2 → map_arterial | +1.23 |
| Norepinephrine ×½ → map_arterial | −0.70 |

Factual NLL increased by **0.00331 nats per reading**, below the 0.005 allowance on tuning.

These are expected model readouts, not established causal truth. Norepinephrine’s magnitude remains uncertain.