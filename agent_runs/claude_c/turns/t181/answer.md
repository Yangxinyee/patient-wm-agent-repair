Last result: run 134 only listed the latent-path tensor names; nothing new was learned. Pass 23 most likely tested the 48 query-sharpening ops on head 0, which are still active. Next I will read the short summary of that pass to decide whether to keep or drop them.

```python
import os, json, pandas as pd, numpy as np
d='/work/fwd_23'
fs=sorted(os.listdir(d)); print(fs)
for f in fs:
    p=os.path.join(d,f)
    if f.endswith('.json'):
        s=open(p).read(); print(f, s[:600])
    elif f.endswith('.csv'):
        df=pd.read_csv(p); print(f, df.shape, list(df.columns)[:10])
print(len(ops()))
```