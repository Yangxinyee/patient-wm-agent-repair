The reconstructed posterior matches closely, but the small local dose responses are model predictions—not evidence of the true effect. I’ll recover the prepared features and edit helpers, then test whether a dose-sensitive correction can meet the record-derived insulin target without harming factual fit. Failure on held-out factual predictions would rule out that correction.

```python
import inspect, numpy as np, pandas as pd
print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())
print("\nLATEST NOTES")
print(open("/work/notes.md").read()[-14000:])
print("\nKERNEL INVENTORY")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    obj = globals()[key]
    if isinstance(obj, np.ndarray):
        print(key, "array", obj.shape, obj.dtype)
    elif isinstance(obj, pd.DataFrame):
        print(key, "dataframe", obj.shape, list(obj.columns)[:12])
    elif isinstance(obj, dict):
        print(key, "dict", list(obj.keys())[:18])
    elif inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function", inspect.signature(obj))
```
