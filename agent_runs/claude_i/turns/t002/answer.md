Nothing exists yet, so this is a fresh start. The architecture tells me where dose information enters the model. Insulin and norepinephrine atoms go through `embed.*`, which uses the z-scores and fine bins of rate and dose, then through `encode_action.*` to give u_k. From there they feed into `z_post`, the prelude, and the recurrent core, and the core's state feeds the value heads. The evaluation doubles or halves rate and dose together.

Before I can judge what the model does wrong, I need to know what the records look like:
- the event mix;
- the units, especially whether more than one rate_unit or dose_unit appears;
- dose distributions and the dose = rate × 1 h relation;
- timing.

I'll start the notebook now so a restart doesn't lose context.

```python
import numpy as np, pandas as pd, json, time
t0=time.time()
train = pd.read_csv('/data/train.csv')
tune = pd.read_csv('/data/tuning.csv')
print(train.shape, tune.shape, time.time()-t0)
print(train.head(30).to_string())
print(train.dtypes)
print(train.event.value_counts())
for c in ['rate_unit','dose_unit']:
    print(train.groupby('event')[c].value_counts(dropna=False))
print(train.stay_id.nunique(), tune.stay_id.nunique())
with open('/work/notes.md','w') as f:
    f.write("# Notebook\n\n## Session 1\n- Read ARCHITECTURE.md and README.md. Loaded train (30k stays) and tuning (400).\n")
```