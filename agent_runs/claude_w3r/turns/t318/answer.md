The rollout showed my core channel does not deliver a rate-dependent signal to the value head: edited insulin shifts are exactly 0.0 (both x2 and x1/2) and norepinephrine shifts are near-equal (−0.31 vs −0.39) despite opposite dose direction. My local test (`QG + DQ['ins']`) worked, but the real rollout's q is not tracking rate the way DQ predicts. Most likely cause: the b_in rank-1 writes a signal of ~10 along a y-direction, but out_norm plus the chained time-flow dilute it before it reaches q.

Repair plan: bypass the fragile core channel. The original model already propagates some rate information into `out` (the original predicts insulin x2 → −0.52). I will fit, from fwd_2's existing states (SX, 5301 bundles across many stays), a direction in `out`-space that linearly predicts the current insulin and norepinephrine rate. I then replace my value-head rate readers (RR) with those directions. The gf/map gate and the bin-centre tilt stay the same; only the read direction changes. The core edits can stay (they do no harm when the read is inert along their delta), but will be revisited.

Outcome that confirms: a direction in `out` predicts rate with decent R² (>0.3 for insulin at drug bundles). If not, I need a different source.

```python
import numpy as np, os, pandas as pd, pickle
G = globals()
for n in ['SX','SST','SHR','UX','YXi','YXn','pa','U_act','RR','TY','WG','GD']:
    if n in G: v=G[n]; print(n, type(v).__name__, getattr(v,'shape',None))
p='/work/fwd_2/states_original.npz'
print('exists', os.path.exists(p))
if os.path.exists(p):
    z=np.load(p); print('keys:', list(z.files))
    for k in z.files: v=z[k]; print(' ',k,v.shape,v.dtype, (v.min(),v.max()) if np.issubdtype(v.dtype,np.number) and v.size<2e7 else '')
try:
    pn=pickle.load(open('/work/panel.pkl','rb'))
    if isinstance(pn,dict):
        for k,v in pn.items(): print('panel',k,type(v).__name__,getattr(v,'shape',None))
    else: print('panel', type(pn).__name__)
except Exception as e: print('pn err',e)
tr=pd.read_csv('/data/train.csv', nrows=3); print('tr cols:', tr.columns.tolist()); print(tr)
```