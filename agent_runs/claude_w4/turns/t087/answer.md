Run 78 showed two things. My CPU replica of the z-posterior mean matches the model to 6e-5 in its first half. The second half does not match, probably because it is the log-variance and goes through a different transform. Several hidden units of the z_post MLP are inactive at drug bundles (importance 0), so they are free to be repurposed. No edits are saved yet.

After the restart I need the exact architecture again and an inventory of the kernel variables before building the injection edit. So this run only reads.

```python
import os, json, numpy as np
print(open('/model/ARCHITECTURE.md').read())
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and k not in ('In','Out','exit','quit','get_ipython')}
items = []
for k, t in g.items():
    v = globals()[k]
    if hasattr(v, 'shape'):
        items.append(f"{k}:{t}{tuple(v.shape)}")
    elif t in ('function','module','type','builtin_function_or_method'):
        items.append(f"{k}:{t[:4]}")
    else:
        try:
            items.append(f"{k}:{t}[{len(v)}]")
        except Exception:
            items.append(f"{k}:{t}")
print(' | '.join(items))
```