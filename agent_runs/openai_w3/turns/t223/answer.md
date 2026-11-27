The carrier-only decoder preserves insulin changes fairly well but still understates norepinephrine changes. Reading unrelated latent features may also contribute to the current rollout drift. I’ll recover the transport construction before changing it; a carrier-only edit would be promising if it reduces factual drift while retaining bidirectional dose responses.

```python
import inspect, numpy as np, torch, os
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== SELECTED RECENT OBJECTS ===")
for _key200 in sorted(list(globals())):
    if any(_key200.endswith(str(n)) for n in [190,191,192,193,198,199]):
        _v200 = globals()[_key200]
        if isinstance(_v200, dict):
            print(_key200, "dict", list(_v200.keys())[:16])
        elif isinstance(_v200, (np.ndarray, torch.Tensor)):
            print(_key200, type(_v200).__name__, tuple(_v200.shape))
        elif callable(_v200):
            try: print(_key200, inspect.signature(_v200))
            except Exception: print(_key200, "callable")
        elif isinstance(_v200, (list, tuple)):
            print(_key200, type(_v200).__name__, "len", len(_v200))
        else:
            print(_key200, type(_v200).__name__, str(_v200)[:100])
print("\n=== REPLAY / INSTALL HELPERS ===")
for _key200 in sorted(list(globals())):
    if callable(globals()[_key200]) and any(s in _key200.lower() for s in
        ["transport","install","carrier","latent","prelude","checkpoint","decode"]):
        try: print(_key200, inspect.signature(globals()[_key200]))
        except Exception: pass
```
