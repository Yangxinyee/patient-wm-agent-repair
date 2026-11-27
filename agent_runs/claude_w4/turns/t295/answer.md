The last run fitted ridge readouts from the model state to its predicted mean for each reading type. Fit was very good: glucose_finger CV R² 0.99, glucose_lab 0.99, map_cuff 0.97, and map_arterial 0.96 with heavy regularisation. The state direction that moves predicted finger glucose by 1 mg/dL has norm about 1.2. It is only partly aligned with the "news" direction that a surprising reading writes into the state (cos 0.56). So a reading surprise moves more than the predicted level. In the example stay, infusions are charted 5 minutes after each hour's readings, so a dose can only show up in later hours.

My notebook ends with "v4(1,1) built and saved; next: dose_edit". Forward passes 1–8 have been used, but fwd_8 is not described in the notes. Python runs 262–265 are also unnoted. Before spending a forward pass, I need to recover three things:
- **fwd_8:** what it contained (it may already be the v4 dose_edit test).
- **True effects:** how I estimated them from the records, which sit in the early part of the notebook. Truth on sel100 is ins x2 −12.96 and NE x2 +4.94, and the x½ values are exactly −½ of these, which suggests a linear dose effect.
- **Kernel state:** whether build_v4 and its parts are still in memory.

What would change my plan: if fwd_8 is already the v4 dose_edit, I calibrate the gains from it now and skip a forward pass. If not, the next step is a dose_edit "both" run on sel100.

```python
import os, glob, json
import pandas as pd, numpy as np
txt = open('/work/notes.md').read()
print('notes length', len(txt))
# print the beginning (causal analysis / truth) compactly
print(txt[:6500])
print('=====')
for d in sorted(glob.glob('/work/fwd_*'), key=lambda s:int(s.split('_')[-1])):
    print(d, sorted(os.listdir(d)))
d8 = '/work/fwd_8'
for f in sorted(os.listdir(d8)):
    p = os.path.join(d8, f)
    if f.endswith('.csv'):
        df = pd.read_csv(p); print(f, df.shape); print(df.head(3).to_string()); 
        if 'edit' in df.columns or any('shift' in c for c in df.columns):
            print(df.describe().T.head(12).to_string())
    elif f.endswith('.json') or f.endswith('.txt') or f.endswith('.md'):
        print(f, open(p).read()[:1500])
print('kernel has:', [k for k in ['build_v4','build_v3','sel100','qs','TA','step_full'] if k in globals()])
```