I now have the full tensor list. The dose pathway is narrow:
1. Insulin and norepinephrine rate/dose enter through `embed.numeric` (z_dose and z_rate columns), `embed.bins.dose` and `embed.bins.rate`.
2. Those atom vectors go through `encode_action.*` to give u_k.
3. u_k enters the core twice: the prelude tokens, and z_post's input [ctx, e_k, u_k] (576 = 3x192).

`last_numeric` takes 8 inputs, not the 4 the document listed, so I'll need to check that later.

Next I'll load train.csv and tuning.csv and describe them:
- event counts and the number of events per stay;
- how long stays last;
- rate distributions for each drug;
- the timing of measurements relative to infusions.

That tells me what data I have for estimating the true causal effects.

```python
import pandas as pd, numpy as np
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.groupby('stay_id').size().describe())
print(tr.groupby('stay_id').hour.max().describe())
```