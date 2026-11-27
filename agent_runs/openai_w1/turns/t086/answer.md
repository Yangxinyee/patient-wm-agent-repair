The inventory was cut off by a large lookup table, but it confirmed that no weights are edited. I’ll inspect only the candidate fits and reconstruction helpers. This should reveal whether the mismatch comes from the encoders, context, or posterior activation, rather than from the statistical fit.

```python
import inspect
import dis
import io
import numpy as np
import torch

print("=== CANDIDATE OBJECTS ===")
for _prefix in ("rr50", "rr51", "rr58", "r73", "r74"):
    print("PREFIX", _prefix)
    for _n in sorted(n for n in list(globals()) if n.startswith(_prefix)):
        _v = globals()[_n]
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_n, type(_v).__name__, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_n, "dict", len(_v), "keys", [str(k)[:45] for k in list(_v)[:8]])
        elif isinstance(_v, (list, tuple)):
            print(_n, type(_v).__name__, len(_v))
        elif callable(_v):
            print(_n, "function")
        else:
            print(_n, str(_v)[:120])
print("\n=== OTHER CACHE NAMES ===")
print([n for n in sorted(list(globals())) if any(s in n.lower() for s in
      ("cache", "posterior", "atom", "encoder", "latent", "direction", "residual"))][:90])

print("\n=== HELPER IMPLEMENTATIONS ===")
for _name in ("r34_activation", "r34_mlp", "r34_set_encode", "r36_atom"):
    _fun = globals().get(_name)
    if not callable(_fun):
        continue
    print("\nFUNCTION", _name)
    try:
        print(inspect.getsource(_fun)[:2600])
    except (OSError, TypeError):
        _buf = io.StringIO()
        dis.dis(_fun, file=_buf)
        print(_buf.getvalue()[:2700])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 77\nNo edit is applied. Narrowing retained-object inspection and recovering encoder helper implementations to diagnose posterior mismatch. Counts: 77 Python runs, 6 forward passes.\n")
```