Run 173 found no unsaved notes. `notes_add` is just Session 6's start, and `src` is the tool library, so the run-170 probe has to be inferred from the variables. That run printed bundle counts of 9918 (about 100 stays) and probed per condition, using bundles from hour 12 on (n = 276 insulin, 480 norepinephrine). My guess is that it re-ran my local core step with the edited doses and checked that a rate probe still holds. Next I'll list array and tensor shapes to find those arrays, with an `isinstance` filter this time. If arrays of length 9918 for E, out or u turn up, the guess is confirmed. Otherwise I'll rebuild the probe.

```python
import numpy as np, torch
items=[]
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)) and v.ndim>0:
        items.append((k,tuple(v.shape)))
print(len(items))
print(' '.join(f"{k}{s}" for k,s in sorted(items)))
for k in ['gain','ins_k','kappa','lam','mult','k_','c12','c13','sl','rho','b1_','b2_','h','lb','tau_min','decay','sp','sp_','abar','ab_']:
    v=globals().get(k)
    if v is None: continue
    if isinstance(v,(np.ndarray,torch.Tensor)) and v.size if isinstance(v,np.ndarray) else (isinstance(v,torch.Tensor) and v.numel()<=8):
        print(k, np.round(np.asarray(v).ravel()[:8],4))
    elif not isinstance(v,(np.ndarray,torch.Tensor)): print(k, v)
```