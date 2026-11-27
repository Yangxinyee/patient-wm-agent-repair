Real progress:
- My fresh prelude replica reproduces the stored prelude outputs exactly.
- `d_idx` maps the 2319 drug bundles exactly onto the GPU state rows.
- `step_full` runs and returns new slots and `out`.

The last error was mine again: `r2` had been overwritten by an old dict in the kernel. I'll name the helper `r2f`.

Next I'll:
1. Score `step_full` against the exact states.
2. Time one call.
3. Check whether it can run on a subset of rows; that matters for computing Jacobians cheaply.
4. Count how many drug bundles are directly followed by a finger or arterial‑MAP reading. These are the readings whose prediction depends only on the drug‑bundle state.

R² ≥ 0.99, with a matching subset call, means the CPU chain is ready for the injection Jacobian next run.

```python
import time
def r2f(a,b): return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()
yv=A(r[0]); ov=A(r[1]); print('R2 y %.5f out %.5f'%(r2f(yv,yt_),r2f(ov,ot_)), ' rel err y %.4f out %.4f'%(np.linalg.norm(yv-yt_)/np.linalg.norm(yt_),np.linalg.norm(ov-ot_)/np.linalg.norm(ot_)))
Ht=torch.tensor(A(htf),dtype=torch.float32); t0=time.time()
r_=step_full(np.arange(2319),torch.tensor(Ep,dtype=torch.float32),Ht); print('time full %.2fs'%(time.time()-t0))
try:
    sub=np.array([3,50,700,1500,2318]); rs=step_full(sub,torch.tensor(Ep[sub],dtype=torch.float32),Ht[sub])
    print('subset ok, max abs diff y %.2e out %.2e'%(np.abs(A(rs[0])-yv[sub]).max(),np.abs(A(rs[1])-ov[sub]).max()))
except Exception as ex: print('subset err',repr(ex)[:300])
pos={int(s):k for k,s in enumerate(dix)}
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    m=(fc.event==ev)&fc.bi.isin(list(pos.keys())); print(ev,'readings right after a drug bundle:',int(m.sum()),'of',int((fc.event==ev).sum()))
print('drug bundle types: ins',int(hi[dix].sum()),'ne',int(hn[dix].sum()),'both',int((hi[dix]&hn[dix]).sum()))
```