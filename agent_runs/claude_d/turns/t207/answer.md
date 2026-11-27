The last run only listed the shapes of the z-posterior and z-prior tensors. It looks like I was starting to examine the z pathway. The notebook says the gated-channel head edit (E2) was saved for a dose_edit test in forward pass 8, but it never records that result. Before doing anything else I need to recover what fwd 8 showed.

What would change my plan:
- If the edited model's insulin x2 shift is still far smaller than the target of about -14, then a head-only edit is too weak. That would explain why I moved on to the z pathway.
- If the shift is close to the target, the remaining problem is only the factual-NLL budget: the offline estimate was +0.0063, over the 0.005 limit.

```python
import os, json, glob
import pandas as pd
print(sorted(os.listdir('/work/fwd_8')))
for f in sorted(glob.glob('/work/fwd_8/*')):
    if f.endswith('.csv'):
        d = pd.read_csv(f)
        print(f, d.shape, list(d.columns)[:12])
        print(d.head(3).to_string()[:800])
    elif f.endswith('.json'):
        print(f, open(f).read()[:1500])
    elif f.endswith('.txt') or f.endswith('.md'):
        print(f, open(f).read()[:1500])
n = open('/work/notes.md').read()
print(len(n)); print(n[-1500:])
print(os.listdir('/work/edited') if os.path.isdir('/work/edited') else 'no dir')
```