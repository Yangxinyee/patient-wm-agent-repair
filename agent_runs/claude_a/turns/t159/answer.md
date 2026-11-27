The last output was blocked, so I lost the result of run 153. My notebook stops at run 142, which means runs 143–153 are not recorded. No edits are saved right now.

Before building the edit, I need to see what is still in the kernel. That covers functions, fitted arrays and the dim-cost file (`dimcost.npy`). I'll print a short, truncated summary only.

Hypothesis: the offline replica functions are still defined (`rollout_stay`, `tf_eval`, `core_step_b`, `head_value`, `make_atoms`, `encode_bundles`). I also expect partial write/readout fit results from runs 143–153.
- **Confirmed** if those names appear.
- **Refuted** if the kernel was reset; then I would need to rebuild from notes.

```python
import types, inspect, numpy as np, torch, os, json
g = {k:v for k,v in list(globals().items()) if not k.startswith('_')}
funcs, arrs, small = [], [], []
for k,v in sorted(g.items()):
    if isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'): continue
    if isinstance(v, np.ndarray): arrs.append(f"{k}{v.shape}")
    elif torch.is_tensor(v): arrs.append(f"{k}T{tuple(v.shape)}")
    elif callable(v) and not isinstance(v,type):
        try: funcs.append(f"{k}{str(inspect.signature(v))[:40]}")
        except Exception: funcs.append(k)
    elif isinstance(v,(int,float)): small.append(f"{k}={v:.4g}" if isinstance(v,float) else f"{k}={v}")
    elif isinstance(v,(list,tuple,dict)): small.append(f"{k}:{type(v).__name__}{len(v)}")
    else: small.append(f"{k}:{type(v).__name__}")
print("N funcs",len(funcs)); print(" | ".join(funcs)[:2500])
print("N arrays",len(arrs)); print(" | ".join(arrs)[:1800])
print("N other",len(small)); print(" | ".join(small)[:1500])
dc=np.load('/work/dimcost.npy'); print('dimcost',dc.shape, np.round(np.sort(dc.ravel())[:8],4))
print(os.listdir('/work/edited')[:10])
```