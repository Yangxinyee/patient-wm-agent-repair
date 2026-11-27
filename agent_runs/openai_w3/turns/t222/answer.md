```python
import inspect, numpy as np, torch, os
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== RECENT OBJECTS ===")
for key in sorted(list(globals())):
    if any(key.endswith(str(n)) for n in range(187,200)):
        v = globals()[key]
        if isinstance(v, dict):
            print(key, "dict", list(v.keys())[:22])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(key, type(v).__name__, tuple(v.shape))
        elif callable(v):
            try: print(key, inspect.signature(v))
            except Exception: print(key, "callable")
        elif isinstance(v, (list, tuple)):
            print(key, type(v).__name__, "len", len(v))
        else:
            print(key, type(v).__name__, str(v)[:160])
print("\n=== REPLAY / INSTALL HELPERS ===")
for key in sorted(list(globals())):
    if callable(globals()[key]) and any(s in key.lower() for s in
        ["transport","install","carrier","latent","prelude","checkpoint","decode"]):
        try: print(key, inspect.signature(globals()[key]))
        except Exception: pass
print("\n=== LAST NOTES ===")
print(open("/work/notes.md").read()[-5000:])
```
