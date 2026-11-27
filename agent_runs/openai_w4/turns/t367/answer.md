FINAL ANSWER

I retain the ten edited tensors in bundle input, prelude attention, and latent-state projections, calibrated toward record-derived dose effects without gradients. I made no further, unvalidated changes.

Expected readouts, anchored to the final validation rollout:

| Dose edit | Predicted mean shift |
|---|---:|
| Insulin ×2 → glucose_finger | −11.678 |
| Insulin ×½ → glucose_finger | +5.587 |
| Norepinephrine ×2 → map_arterial | +2.194 |
| Norepinephrine ×½ → map_arterial | −10.257 |

Factual NLL penalties were **+0.00322** and **+0.00469 nats** on two tuning cohorts—both within the limit, though held-out compliance remains uncertain.

This is an incomplete repair. Norepinephrine halving generalized poorly: its predicted shift changed from −3.96 to −10.26 between cohorts, versus the record-derived estimate of approximately −2.43. These rollout predictions are not established simulator truth.