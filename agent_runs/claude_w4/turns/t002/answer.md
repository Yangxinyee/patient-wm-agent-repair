## Step 1: first look at the model and the data

**What the last result showed.**
- Dose information reaches glucose and MAP predictions only through the action encoder. Each insulin or norepinephrine atom carries z_rate, z_dose, has_rate and has_dose (through `embed.numeric`), plus rate and dose bins (through `embed.bins.rate` and `embed.bins.dose`). These pass through `encode_action` to give u_k.
- u_k then enters the latent posterior and the prelude, and from there the recurrent state.
- The "last value" registers only carry same-code history. A glucose prediction therefore never sees a drug rate directly; it sees it only through the state.
- There is no notebook yet, so this is a fresh start.

**What this means.** The model's dose effect equals the partial association between dose and later readings given the history that it learned in training. If clinicians dosed on information that is not in the record, that association is confounded. In that case I must find variation in dose that is as good as random, for example:
- protocol thresholds or discrete dose steps (a regression-discontinuity design);
- unit or recording quirks;
- stay-level clinician habits;
- noise that does not depend on the patient's state.

**What I will do next.** I will start the notebook, then look at:
- the shape of the records;
- the vocabulary (codes, units, normalisation, bins);
- the tensor list;
- basic dosing patterns (rate distributions, units, timing).

**What would count as evidence.** If rates show discrete steps tied to the last reading (a sliding scale), that points to a discontinuity design. If they vary smoothly with noise, I will need some other source of exogenous variation.

```python
import numpy as np, pandas as pd, json, os, time
os.makedirs('/work', exist_ok=True)
with open('/work/notes.md','w') as f:
    f.write("# Notes\n\n## Run 1-2: read ARCHITECTURE.md, README. Dose enters only via encode_action (z_rate,z_dose,bins) -> u_k -> z_post & prelude.\n")
t0=time.time()
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape, time.time()-t0)
print(tr.head(30).to_string())
print(tr.dtypes)
print(tr.event.value_counts())
print(tr.groupby('event')[['rate_unit','dose_unit']].agg(lambda s: s.value_counts(dropna=False).to_dict()))
voc = VOCAB
print(voc.keys())
print(json.dumps({k:(v if not isinstance(v,(list,dict)) or len(str(v))<600 else str(v)[:600]) for k,v in voc.items()}, indent=1)[:6000])
```