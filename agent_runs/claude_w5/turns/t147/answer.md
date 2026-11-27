The last output was blocked, so I don't know what runs 129–135 did. The notebook ends at the plan: check the head locally against fwd_3, then build the dose edit. Before writing new code I'll take a short inventory of the kernel: which functions and variables exist, and their signatures. That tells me whether a local head implementation was already built after the last notebook entry. If a head function exists and matches fwd_3 predicted means, I move to designing the edit. If not, building the head is the next step.

```python
import inspect, types
g = dict(globals())
fns, arrs, others = [], [], []
for k, v in g.items():
    if k.startswith('_') or k in ('In','Out','exit','quit','get_ipython'):
        continue
    if isinstance(v, types.ModuleType):
        continue
    if callable(v) and not isinstance(v, type):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = '?'
        fns.append(f"{k}{sig[:80]}")
    else:
        try:
            import numpy as np, torch
            if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                arrs.append(f"{k}:{tuple(v.shape)}")
            elif isinstance(v, (list, dict, tuple, set)):
                others.append(f"{k}:{type(v).__name__}[{len(v)}]")
            elif isinstance(v, (int, float, str, bool)):
                s = repr(v)
                others.append(f"{k}={s[:40]}")
            else:
                others.append(f"{k}:{type(v).__name__}")
        except Exception as e:
            others.append(f"{k}:err")
print("FUNCTIONS:", len(fns))
for f in fns: print("  ", f)
print("ARRAYS:", ", ".join(arrs)[:1500])
print("OTHERS:", ", ".join(others)[:2500])
import os
print(os.listdir('/work'))
for d in ['fwd_1','fwd_2','fwd_3']:
    print(d, os.listdir('/work/'+d))
```