The kernel still holds the tool's states for the 5156 bundles of fwd_3 (X3 = [out, 4 h-slots, z] after each bundle) and the per-bundle stay and hour (sid3, hr3). It also has `atoms_by_bundle`, which feeds my exact encoder `eu2`, and `core_step`. Before the leverage experiment I need one self-consistent pipeline, so this run rebuilds e and u for every bundle with `eu2`. It then checks two things on 60 random bundles: the z posterior recomputed from ctx, and `core_step`'s new state against the tool states. Small errors (median relative error about 1e-6, as in my notes) mean I can trust the local simulation for designing the edit. Large errors would mean a wrong argument convention, which I would fix first.

```python
import time
T=torch.float64
sid=np.asarray(sid3); hr=np.asarray(hr3,dtype=float); tm=hr*60.0
X3t=torch.tensor(np.asarray(X3),dtype=T)
first=np.r_[True, sid[1:]!=sid[:-1]]; j0_of=np.maximum.accumulate(np.where(first,np.arange(len(sid)),0))
print('WT dtype',WT['core.flow.a'].dtype, 'n stays',first.sum())
t0=time.time()
try:
    r=eu2(atoms_by_bundle[1]); print('eu2 ->',type(r),[tuple(x.shape) for x in r])
except Exception as ex: print('eu2 err',repr(ex)[:300]); print(type(atoms_by_bundle[1]), str(atoms_by_bundle[1])[:200])
```