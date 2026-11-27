Many arrays from earlier analyses are still in the kernel, but their meanings aren't recorded, so I'd rather recompute from states files. `core.b_in.weight` is 192x192 and maps E into the per-slot h space, so an output-side rank-1 op along an h direction is possible. Next I'll find which earlier passes saved states files, and whether they hold both the original and an edited model. With those I can estimate the insulin dose imprint in h at drug bundles and compare it with the reading-bundle step variation along the same direction.

```python
sf=sorted(glob.glob('/work/fwd_*/states_*.npz')); print(sf)
z=np.load(sf[0], allow_pickle=True); print({k:z[k].shape for k in z.files})
```