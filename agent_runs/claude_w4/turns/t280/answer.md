I'm starting over after a kernel crash, and runs 205–251 left no notes. First I need to recover what fwd_6 and fwd_7 showed and what the current edit is. The edit lives in `core.prelude.0.attn.kv/out`, which matches the v3 head-1 carrier. I'll list /work, read the fwd_6/fwd_7 outputs, look inside `v3_g1_1_0.npz`, and measure how the current edit differs from the original. I'll keep memory low, since the last run died from memory pressure. If fwd_6 shows a nonzero insulin and NE dose response for v3(1,1,0), that confirms the head-1 value path is a working carrier and only needs gain calibration. If it is again identical to the original, the carrier is wrong.

```python
import os, json, numpy as np, pandas as pd
print("edited:", edited())
for root, dirs, files in sorted(os.walk('/work')):
    for f in sorted(files):
        p=os.path.join(root,f); print(p, os.path.getsize(p))
txt=open('/work/notes.md').read(); print("notes len", len(txt))
for k in [6,7]:
    d=f'/work/fwd_{k}'
    if not os.path.isdir(d): continue
    for f in sorted(os.listdir(d)):
        p=os.path.join(d,f)
        if f.endswith(('.json','.txt','.md')):
            s=open(p).read(); print('=====',p,'\n',s[:3000])
        elif f.endswith('.csv'):
            df=pd.read_csv(p); print('=====',p,df.shape); print(df.columns.tolist()); print(df.head(3).to_string())
            for c in ['edit','event','variable']:
                if c in df.columns: print(df[c].value_counts().head(10))
        elif f.endswith('.npz'):
            z=np.load(p); print('=====',p,{kk:z[kk].shape for kk in z.files})
z=np.load('/work/v3_g1_1_0.npz'); print('v3 npz', {kk:(z[kk].shape, float(np.abs(z[kk]).sum())) for kk in z.files})
for n in edited():
    a=np.asarray(load(n)); b=np.asarray(load_original(n)); d=a-b
    rows=np.where(np.abs(d).sum(1)>0)[0]; cols=np.where(np.abs(d).sum(0)>0)[0]
    print(n, a.shape, 'diffnorm', float(np.linalg.norm(d)), 'rows', (rows.min(), rows.max(), len(rows)) if len(rows) else None, 'cols', (cols.min(), cols.max(), len(cols)) if len(cols) else None)
    for kk in z.files:
        if z[kk].shape==a.shape: print('  matches npz', kk, float(np.abs(z[kk]-a).max()))
```