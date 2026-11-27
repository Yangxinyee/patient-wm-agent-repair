mlp1 is Linear(192→768) → GELU → Linear(768→192) with no norm before it. So a gate on "NE code AND bin 16" can be built exactly from the additive atom embedding, and it would not touch any atom that appears in the records. fwd11's per-stay-hour shifts are available with a `reached` flag.

Next I will split the ne_x2 stay-hours by whether the preceding doubled NE rates went above 1.0. Two readings of the outcome:
- If out-of-range hours show about zero shift, the gate is failing (explanation a).
- If they show a flat plateau regardless of dose, the decoded dose is saturating (explanation b).

This run first checks that I can reproduce the per-hour targets and finds the record columns needed.

```python
import json
for f in ['dose_panel','hourly_panel','stay_panel','prev_dose','panel','true_shift']:
    c=globals()[f].__code__; print(f, c.co_varnames[:c.co_argcount], globals()[f].__defaults__)
TU=pd.read_csv('/data/tuning.csv'); print(TU.shape, list(TU.columns)); print(TU[TU.iloc[:,1].astype(str).str.contains('norep')].head(3).to_string() if TU.shape[1]>1 else '')
gam=json.load(open('/work/effects.json'))['gam_ne_clean']
E=D11[(D11.edit=='ne_x2')&D11.reached].copy()
try:
    t=[true_shift_ne(s,h,2.0,gam) for s,h in zip(E.stay_id.values[:5],E.hour.values[:5])]; print('tsne', t)
except Exception as ex: print('ERR', repr(ex)[:300])
```