The last few replies were blocked, so I will rebuild context with short outputs only. I'll check what the kernel still holds (functions and arrays from the exact local replica) and what `/work/samp_pert.pkl` contains. This tells me whether I can resume the plan in my notes: decode the dose innovation from states built with randomly perturbed doses, then add a head-level tilt. If the replica functions (`run_stay`, `core_step`, the head replica) are still defined, I can continue. If not, I'll need to rebuild them from my notes.

```python
import pickle, os, types, numpy as np
g = {k: v for k, v in globals().items() if not k.startswith('_')}
funcs = sorted([k for k, v in g.items() if isinstance(v, types.FunctionType)])
arrs = sorted([(k, tuple(v.shape)) for k, v in g.items() if isinstance(v, np.ndarray) and v.size > 1000], key=lambda x: -np.prod(x[1]))[:25]
print("n funcs", len(funcs)); print(", ".join(funcs)[:1500])
print("big arrays:", arrs)
others = sorted([k for k, v in g.items() if isinstance(v, (dict, list, tuple)) and k not in ('NAMES','SHAPES','VOCAB')])
print("containers:", ", ".join(others)[:800])
try:
    sp = pickle.load(open('/work/samp_pert.pkl', 'rb'))
    print("samp_pert type", type(sp))
    if isinstance(sp, dict):
        for k, v in list(sp.items())[:15]:
            print(" ", k, type(v).__name__, getattr(v, 'shape', (len(v) if hasattr(v, '__len__') else v)))
except Exception as e:
    print("pkl err", e)
```