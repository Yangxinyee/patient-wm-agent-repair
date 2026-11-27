My kernel already has, for the 5,156 bundles of the fwd_3 stays, the encoder outputs (`euF`: e and u per bundle) and the tool's true states (`X3`: out, the 4 h slots and z per bundle). That is enough for a one-step check of the core:
1. Take the tool's state after bundle k-1.
2. Apply my implementation of the time flow, prelude, recurrent blocks and coda.
3. Compare with the tool's state after bundle k.

The architecture text leaves a few details open:
- the time unit of the memory "age";
- whether the memory includes the current bundle;
- how the age embedding enters the keys.

I will test these as variants. First I need to confirm the per-bundle timing arrays I built earlier (`dt_s`, `min_`). Next I will print a few entries next to the hours and stay ids, and cache all weights as torch tensors for the core implementation.

This step only prepares data, so there is no hypothesis to test yet.

```python
import torch
torch.set_grad_enabled(False)
WT = {n_: torch.tensor(load(n_), dtype=torch.float32) for n_ in NAMES}
print('dt_s', tuple(dt_s.shape), np.round(dt_s[:10].numpy(),1))
print('min_', tuple(min_.shape), np.round(min_[:10].numpy(),2))
print('prev_ok', tuple(prev_ok.shape), prev_ok[:10].numpy())
print('H3', np.round(H3[:10],3), 'S3', S3[:3], 'natoms', z3['n_atoms'][:10])
print('hours monotone within stay:', all(np.all(np.diff(H3[S3==s_])>=0) for s_ in np.unique(S3)[:20]))
print('X3 first row h-slot norms', np.round(np.linalg.norm(X3[0,192:960].reshape(4,192),axis=1),2), 'out norm', np.round(np.linalg.norm(X3[0,:192]),2))
```