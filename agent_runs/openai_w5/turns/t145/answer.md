The earlier checks show similar insulin slopes across informative dose ranges, but norepinephrine effects above the recorded cap remain extrapolations. The helper inventory was again truncated by large default arguments. I’ll retrieve only the recent channel objects and exact validation IDs, then test factual fit. A paired NLL increase above 0.005 would reject this transport candidate before any effect correction.

```python
import inspect
import numpy as np

print("=== VALIDATION IDS ===")
print("A:", list(eval_ids_A))
print("B:", list(eval_ids_B))

print("=== EFFECT SUMMARIES ===")
for _n in ("dyn_estimates", "dynamic_alternatives", "sensor_checks", "best_head_spec"):
    if _n in globals():
        print(_n, repr(globals()[_n])[:1000])

print("=== RECENT GLOBAL NAMES ===")
_recent = [(_n, _v) for _n, _v in list(globals().items())
           if not _n.startswith("_")][-150:]
print(", ".join(_n + (str(tuple(_v.shape)) if hasattr(_v, "shape") else "")
                for _n, _v in _recent))

print("=== CHANNEL OBJECTS ===")
for _n, _v in list(globals().items()):
    if _n.startswith("_") or not _n.endswith(("117", "118", "119", "120", "121",
                                            "122", "123", "124", "125")):
        continue
    if inspect.isfunction(_v):
        print(_n, "args:", _v.__code__.co_varnames[:_v.__code__.co_argcount],
              "uses:", _v.__code__.co_names)
    elif isinstance(_v, dict):
        print(_n, "keys:", list(_v)[:30])
    elif hasattr(_v, "shape"):
        print(_n, "shape:", tuple(_v.shape))
    else:
        print(_n, repr(_v)[:140])
```