The core reconstruction is reasonably close in its output vector, but the head still has systematic error. I’ll inspect the actual feature and edit construction rather than assume prior sampling caused the failure. A feature-space or centering mismatch would explain the discrepancy; otherwise I’ll test sampling sensitivity.

```python
import inspect
import numpy as np
import torch

for name in ("reconstruct_reading_head", "repair_tilt"):
    print("\nSOURCE:", name)
    try:
        print(inspect.getsource(globals()[name]))
    except Exception as exc:
        obj = globals()[name]
        print("Source unavailable:", type(exc).__name__)
        print("locals:", obj.__code__.co_varnames)
        print("constants:", obj.__code__.co_consts)

print("\nRELEVANT CACHED VARIABLES")
terms = ("tilt", "contrast", "coef", "offset", "head", "repair_", "target_")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or inspect.isfunction(obj):
        continue
    if any(term in name.lower() for term in terms):
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj)[:12])
        elif isinstance(obj, (str, float, int, tuple)):
            print(name, repr(obj)[:200])

print("\nFEATURE ALIGNMENT")
for name in ("feat0", "Xf"):
    obj = globals().get(name)
    if obj is not None and np.shape(obj) == np.shape(head_hidden):
        print(name, "vs head_hidden RMS:",
              np.sqrt(np.mean((np.asarray(obj)-head_hidden)**2)))

sample_reconstruction = reconstruct_reading_head(head_meta.iloc[:4])
print("\nRECONSTRUCTION OUTPUT KEYS:", list(sample_reconstruction))
for name in ("numbers", "hidden", "probs"):
    arr = sample_reconstruction[name]
    print(name, tuple(arr.shape))
    if name == "numbers":
        print(arr)
print("ACTIVE EDITS:", edited())
```
