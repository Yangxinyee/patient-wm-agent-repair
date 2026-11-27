Run 79 refutes the hypothesis. No categorical field varies with insulin rate: the unit is always "units/hour", the dose unit is always "units", and dose equals rate. So the binary z pattern does not come from a hidden categorical input. I will stop trying to replicate z exactly, as already decided.

The plan is a head-level edit. The glucose value head would get a term in the insulin dose, fed by a probe of the dose read from the state the head already sees, with compensation through the last-glucose-bin rows. I would then calibrate it end-to-end with dose_edit runs.

Before designing it I need two things:
1. The exact head structure: tensor shapes, and whether there is a LayerNorm before the MLP.
2. The factual noise scale per event: residual sd, NLL per reading, and the mix of events. This tells me how much probe noise the 0.005-nat budget allows.

If the glucose residual sd turns out large (≥15 mg/dL), a probe-based term with about 1.5 mg/dL of noise is affordable. If it is small, the approach is risky.

```python
import json, pandas as pd, numpy as np
for n in NAMES:
    if any(s in n for s in ['attr_heads.value','embed_transition','query_norm','type_emb','last_numeric','last_status','z_head','core.gap','core.out_norm','embed.bins.rate','embed.bins.dose']):
        print(n, SHAPES[n])
print(json.load(open('/work/fwd_3/summary.json')))
fo = pd.read_csv('/work/fwd_3/factual_original.csv')
print(fo.columns.tolist()); print(fo.head(3).to_string())
ev = [c for c in fo.columns if fo[c].astype(str).isin(['glucose_finger','map_arterial']).any()][0]
pm = [c for c in fo.columns if 'mean' in c.lower()][0]
vc = [c for c in fo.columns if c in ('value','reading','obs')]
nl = [c for c in fo.columns if 'nll' in c.lower()]
print('using', ev, pm, vc, nl)
if vc:
    fo['res'] = fo[vc[0]] - fo[pm]
    print(fo.groupby(ev).agg(n=('res','size'), res_sd=('res','std'), res_mean=('res','mean'), **{k: (k,'mean') for k in nl}).to_string())
```