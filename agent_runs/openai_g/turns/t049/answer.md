FINAL ANSWER

The final audit passed. I’m submitting edits to `embed.numeric.weight`, `embed.bins.dose.weight`, and `embed.bins.rate.weight`: dose/rate magnitude dependence is removed, with training-average offsets retained. Drug identity, timing, and measurement inputs remain intact.

Record-based checks favor negligible effects, most strongly for insulin. Delayed norepinephrine effects remain uncertain; zero is a conservative choice, not a proven causal truth.

Expected readouts, confirmed in the tuning rollout:
- Insulin ×2: **0 mg/dL**
- Insulin ×½: **0 mg/dL**
- Norepinephrine ×2: **0 mmHg**
- Norepinephrine ×½: **0 mmHg**

Across all 400 tuning stays, factual NLL increased by **0.0000816 nats/reading**, well below the **0.005** limit.