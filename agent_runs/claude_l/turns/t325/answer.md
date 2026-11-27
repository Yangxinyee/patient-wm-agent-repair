The replica supports this route. `bview` applies `z_prior`, a 192→192→64 MLP with GELU, to the base context. It then samples z and adds `z_head(z)` to the view. So an edit to `z_prior`/`z_head` will show up in both offline tools, `evalW` (rollout readouts) and `tfnll` (teacher-forced NLL). Because `z_head` is zeroed, `z_prior` currently has no effect on anything.

The saved data also helps:
- `s1data` holds per-drug-bundle contexts and residuals (`ri`, `rn`, `hi`, `hn`) for batch1.
- `tf_train` holds the drug-step context for 57k train drug bundles.

To fit a decoder I need the view base at the prediction point after each drug bundle, with exactly the replica's conventions. That means the dt units, the bundle structure, and how `tfnll` calls `bview`.

This run reads the `bview` and rollout code and checks the residual statistics. It also tests capturing `bview` inputs during `tfnll` by temporarily wrapping it.
- **Confirms the plan:** the calls are captured and the dt convention is readable.
- **Refutes it:** `tfnll` bypasses the global `bview`, so I would have to rebuild the bases from the states instead.

```python
import torch, numpy as np, time
L=open('/work/wm2.py').read().split('\n'); print('\n'.join(f"{i}:{l[:105]}" for i,l in enumerate(L) if 41<=i<=58))
R=open('/work/wm2_rollout.py').read().split('\n'); print('\n'.join(f"{i}:{l.rstrip()[:92]}" for i,l in enumerate(R) if 15<=i<=72 and l.strip()))
print(type(BUN), len(BUN)); k0=list(BUN.keys())[0] if isinstance(BUN,dict) else 0; b=BUN[k0]; print(type(b), len(b), str(b[0])[:160])
d=np.load('/work/s1data.npz'); print({k:(round(float(d[k].mean()),3),round(float(d[k].std()),3),int((d[k]!=0).sum())) for k in ['ri','rn','hi','hn']})
_bv=bview; CAP=[]
def bview_cap(*a,**kw):
    r=_bv(*a,**kw); CAP.append([tuple(x.shape) if hasattr(x,'shape') else x for x in a[1:4]]); return r
globals()['bview']=bview_cap
try:
    t0=time.time(); r=tfnll(W_cur,[k0]); print('tfnll',str(r)[:150],'calls',len(CAP),CAP[:2],round(time.time()-t0,1))
except Exception as e: print('tf err',str(e)[:150])
globals()['bview']=_bv
print('zh max',float(W_cur['z_head.weight'].abs().max()))
```