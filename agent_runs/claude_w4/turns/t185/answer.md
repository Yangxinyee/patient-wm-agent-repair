The kernel has a full set of replica and readout tools, so I don't need to rebuild anything. One fact decides whether a one-step injection can reproduce the multi-hour insulin effect: how strongly the model carries a glucose shift forward to its next prediction. In the rollout, a lowered prediction at hour t produces a lowered sampled reading, and the model's persistence ρ then carries it on. The steady-state effect of a sustained dose change is roughly g/(1−ρ), with g the immediate effect per unit. Matching the true cumulative effect of about −10.3 per unit with g ≈ −1.83 needs ρ ≈ 0.82.

I'll estimate ρ from the original model's factual predictions (fwd_2): regress the predicted mean at the next finger reading on the previous prediction and the previous surprise. If ρ is around 0.8, a gain near h1 should work. If it is much smaller, the injection would need a persistent component.

```python
import pandas as pd, numpy as np
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.columns.tolist()[:20], len(fa))
print(fa.head(3).to_string()[:800])
```