The notebook's plan was to make the drug inputs magnitude-blind, and three tensors are already edited. Run 37 showed which stay-hours count in the readout: hours 13–24 whenever insulin ran from hour 12 on. I don't remember what forward passes 2 and 3 tested or returned. Before using more forward passes, I'll re-orient: inspect the outputs in fwd_2/fwd_3, see exactly how the current edits differ from the original, and re-read the relevant ARCHITECTURE parts. If fwd_2/3 show the edited model's dose shifts at ~0 with factual NLL within 0.005, the plan is confirmed. If not, I'll need to refine the edit.

```python
import os, json, glob
import numpy as np, pandas as pd
print(open('/work/notes.md').read()[:200])
for d in sorted(glob.glob('/work/fwd_*')):
    print('==', d, os.listdir(d))
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.endswith('.json'):
            txt = open(p).read()
            print(f, txt[:1500])
        elif f.endswith('.csv'):
            df = pd.read_csv(p)
            print(f, df.shape, list(df.columns)[:20])
            print(df.head(3).to_string())
print(open('/work/sel1.json').read()[:500])
print(os.listdir('/work/edited') if os.path.isdir('/work/edited') else 'edited is file')
for n in edited():
    a = load(n); b = load_original(n)
    print(n, a.shape, 'max abs diff', float(np.abs(a-b).max()))
```