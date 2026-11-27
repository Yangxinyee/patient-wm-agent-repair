FINAL ANSWER

Edited only `embed.numeric.weight`, `embed.bins.dose.weight`, and `embed.bins.rate.weight` to remove dose-magnitude sensitivity while preserving drug presence, identity, timing, and measurement inputs.

Adjusted training-record analyses favor near-zero effects in the evaluated window, though small nonzero effects remain possible.

Expected edited-model readouts, confirmed by rollout checks:
- Insulin ×2: **0 mg/dL**
- Insulin ×½: **0 mg/dL**
- Norepinephrine ×2: **0 mmHg**
- Norepinephrine ×½: **0 mmHg**

Across 400 tuning stays, factual NLL increased by **0.000307 nats/reading**, comfortably below the **0.005** limit.