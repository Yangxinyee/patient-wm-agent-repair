This is a fresh start: there are no notes and /work is empty. From the architecture I learned three things.
- Actions (insulin and norepinephrine) enter through `encode_action` → u_k. u_k feeds into z_post and the prelude.
- The value head reads `with_code`, which includes a "last value" register.
- The dose edit changes the insulin atoms' rate and dose, and therefore their z-scores, their bins and the numeric embedding.

So the model's dose response flows through `embed.numeric` (the z_dose and z_rate columns), `embed.bins.dose` and `embed.bins.rate` for the insulin and norepinephrine codes, then through encode_action and the core.

Next I will look at the records: columns, event counts, dose distributions and timing. I will also look at the vocab (normalisation and bins) and the tensor list. This tells me what the data can support for estimating true drug effects. I have no hypothesis to test yet; this is orientation. I will also start the notebook.

```python
import pandas as pd, numpy as np, json
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.groupby('event')[['value','rate','dose']].describe().T.to_string())
V = json.load(open('/model/vocab.json'))
print(V.keys())
for k in V:
    s = json.dumps(V[k])
    print(k, s[:1500])
    print()
with open('/work/notes.md','w') as f:
    f.write("# Notes\n\n## Run 1-2: orientation\n- Read ARCHITECTURE.md. Actions enter via encode_action -> u_k (to z_post and prelude). Dose edit changes insulin/norepi rate & dose z-scores, bins.\n")
```