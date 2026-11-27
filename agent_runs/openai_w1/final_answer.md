FINAL ANSWER

Retained the nine latent/prelude edits and added an analytical rank-two correction to `core.b_in.weight`, based on dose-response estimates from the training records. No gradients were used.

Approximate expected shifts, in record units:

| Intervention | Expected shift |
|---|---:|
| Insulin ×2 → glucose_finger | −3.2 |
| Insulin ×½ → glucose_finger | +1.4 |
| Norepinephrine ×2 → map_arterial | +0.25 |
| Norepinephrine ×½ → map_arterial | −0.45 |

This is a **partial repair**, not demonstrated agreement with the larger, assumption-dependent record estimates. These final rollout shifts remain provisional.

Factual NLL increased by **0.00344 nats** on the validated 100-stay batch, within the limit. Full-cohort factual safety, all-output log loss, and the additional interventions remain unverified.