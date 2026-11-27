Run 38 recovered the architecture and my kernel variables. Dose reaches the model in one place: the action atoms' embedding, via the dose and rate z-scores and bins. These pass through the action encoder (`encode_action`) into the recurrent state, which the glucose and MAP heads read. Any edit therefore has to change how that action signal moves the state, or how the heads read it.

Before designing an edit, I need the reference "true" effects on the same 100 tuning stays (sel1) where I measured the original model. My notebook has data-derived response kernels: insulin, cumulative about −12 mg/dL per u/h-hour; norepinephrine, about +8.6 mmHg per mcg/kg/min-hour. I never turned these into the four readouts.

Next, I will check the data layout, the stored kernels and the earlier forward-tool files. Then I can compute the kernel-implied effect of each dose edit per stay and compare it with the model's −2.01 / +1.16 / +0.37 / −0.06. If the implied insulin x2 effect is several times larger than −2, and the NE x2 effect several mmHg rather than 0.4, the model under-responds and the edit must amplify the dose pathway.

```python
import numpy as np, pandas as pd, os, json
print(open('/data/README.md').read()[:2500])
print(train.columns.tolist()); print(train.head(12).to_string())
print("K_ins", np.round(np.asarray(K_ins,dtype=float),3))
print("K_ne ", np.round(np.asarray(K_ne,dtype=float),3))
for v in ['K_ins_s','K_ne_s','TR','selT','sel1']:
    o=globals()[v]; print(v, type(o), (np.shape(o) if hasattr(o,'__len__') else o))
for k in [1,2,3]:
    print(k, os.listdir(f'/work/fwd_{k}'))
```