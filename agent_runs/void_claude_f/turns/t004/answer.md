The tensor list matters for the edit design. The dose and rate bin tables (`embed.bins.dose`, `embed.bins.rate`, 17×192) and `embed.numeric` (with z_dose and z_rate columns) are shared by insulin and norepinephrine. Code-specific behaviour can only come from interactions with `embed.code` (insulin = 8, norepinephrine = 9) and the categorical unit embeddings inside the action encoder's nonlinearity. So a clean edit for one drug alone will have to work in `encode_action` (or further downstream), not in the shared tables.

Before thinking about edits I need the facts of the records: how often each drug is given, how doses relate to readings, and whether there is dose variation that does not follow the patient's condition. Such variation is what would let me identify the true effects. I will load train.csv and look at its structure and some example stays.

```python
import pandas as pd, numpy as np, time
t0=time.time()
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(time.time()-t0)
print(tr.shape, tu.shape)
print(tr.dtypes)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.groupby('event')[['value','rate','dose']].describe().T.to_string())
```