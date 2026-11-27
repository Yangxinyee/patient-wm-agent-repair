The replica failed on a type mismatch, not a logic problem. `build_mem` writes rows of the global `e_all` into a torch tensor, but `e_all` is apparently a numpy array now. Its global `hr` is also the stay-hour array I reloaded from the same 5,218-bundle states, so it should line up.

Next I'll convert `e_all` to a float32 torch tensor, check that `st` and `hr` have one entry per bundle, and rerun the 64-bundle comparison against the recorded new slots and `out`. The same criteria as before decide whether the replica can be trusted for Jacobians.

```python
import time
print('e_all type', type(e_all), getattr(e_all,'shape',None), '| st', type(st), getattr(st,'shape',None), '| hr', type(hr), getattr(hr,'shape',None), '| L', L)
if isinstance(e_all,np.ndarray): e_all=torch.as_tensor(e_all,dtype=torch.float32)
print('st matches sid:', np.array_equal(np.asarray(st),sid), ' hr matches states hour:', np.allclose(np.asarray(hr),So['hour']))
t0=time.time()
with torch.no_grad():
    E0t=E0 if isinstance(E0,torch.Tensor) else torch.tensor(E0,dtype=torch.float32)
    htft=htf if isinstance(htf,torch.Tensor) else torch.tensor(htf,dtype=torch.float32)
    res=step_full(di[:64], E0t[:64], htft[:64])
print('time %.2fs'%(time.time()-t0), type(res).__name__, [tuple(getattr(r,'shape',())) for r in (res if isinstance(res,(tuple,list)) else [res])])
outs=res if isinstance(res,(tuple,list)) else [res]
yt_=np.asarray(yt[:64]); ot_=np.asarray(ot[:64])
for r in outs:
    r=np.asarray(r)
    if r.shape==yt_.shape: e=r-yt_; print('y: rms err %.3f  R2 %.5f'%(np.sqrt((e**2).mean()), 1-(e**2).sum()/((yt_-yt_.mean(0))**2).sum()))
    if r.shape==ot_.shape: e=r-ot_; print('out: rms err %.4f R2 %.5f'%(np.sqrt((e**2).mean()), 1-(e**2).sum()/((ot_-ot_.mean(0))**2).sum()))
```