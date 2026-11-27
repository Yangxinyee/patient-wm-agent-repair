FINAL ANSWER

Kept candidate C, changing only `core.b_in.weight` through gradient-free, record-derived dose corrections, shrunk to limit factual degradation.

Expected shifts, based on the final tuning rollout:

| Dose edit | Outcome | Shift |
|---|---|---:|
| Insulin ×2 | glucose_finger | −11.017 |
| Insulin ×½ | glucose_finger | +5.130 |
| Norepinephrine ×2 | map_arterial | +0.363 |
| Norepinephrine ×½ | map_arterial | −0.741 |

Factual NLL increased **0.004065 nats/reading** on 200 checked stays, below the 0.005 limit. Unseen-stay performance is not guaranteed.

Insulin responses improved markedly; norepinephrine responses remain smaller than the record-based effect estimates.