`build_mem` shows how the core step's memory works. For each bundle it collects the event tokens (`e_all`) of up to the last 128 bundles of the same stay, including the current one. Their ages in hours go into the age embedding. So readings, and the drug-record "end" events, reach later steps through this memory as well as through the slots. `rws` is simply the bundle's row index in the states arrays (`di` holds the drug-bundle rows: 3, 5, 7, …).

Next I'll run the replica `step_full(di, E0, htf)` on the first 64 drug bundles. I'll compare its output with the recorded new slots (`yt`) and `out` (`ot`) from the forward states, and time it.
- **Replica usable:** R² near 0.998 for the slots and above 0.99 for `out`. Then I can take Jacobians of `out` with respect to an injection added to the head-1 output, using a small custom prelude that adds the perturbation before the FFN.
- **Inputs mis-wired:** a poor match would call for a fix before going further.

```python
import time
t0=time.time()
with torch.no_grad():
    res=step_full(di[:64], E0[:64] if isinstance(E0,torch.Tensor) else torch.tensor(E0[:64]), htf[:64])
print('time %.2fs'%(time.time()-t0), type(res), [getattr(r,'shape',None) for r in (res if isinstance(res,(tuple,list)) else [res])])
outs=res if isinstance(res,(tuple,list)) else [res]
yt_=np.asarray(yt[:64]); ot_=np.asarray(ot[:64])
for r in outs:
    r=np.asarray(r)
    if r.shape==yt_.shape: e=r-yt_; print('y: rms err %.3f  R2 %.5f'%(np.sqrt((e**2).mean()), 1-(e**2).sum()/((yt_-yt_.mean(0))**2).sum()))
    if r.shape==ot_.shape: e=r-ot_; print('out: rms err %.4f R2 %.5f'%(np.sqrt((e**2).mean()), 1-(e**2).sum()/((ot_-ot_.mean(0))**2).sum()))
print('e_all', getattr(globals().get('e_all'),'shape',None), 'L', globals().get('L'), 'st', getattr(globals().get('st'),'shape',None))
```