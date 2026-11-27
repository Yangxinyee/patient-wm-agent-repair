The head-0 ×4 cost is non-selective, so whole-head scaling is a dead end. The only route specific to drug bundles is the u-token entering prelude attention, where it is mixed with the slot, e and z tokens through shared `kv` weights. A selective amplification needs a direction in the norm1 output space that the dose varies but the slot tokens (h-dominated) do not. As a first feasibility check, I'll measure how many dimensions the direction of h actually spans across bundles, using saved state files. A low-dimensional span would leave a large quiet subspace to work in.

```python
import glob
fl = sorted(glob.glob('/work/fwd_*/states_*.npz')); print([f_.split('/')[-2]+'/'+f_.split('/')[-1] for f_ in fl])
z_ = np.load(fl[0], allow_pickle=True); print({k_: z_[k_].shape for k_ in z_.files})
```