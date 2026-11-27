The architecture confirms that the value head reads a 192-dimensional summary, while drug information enters through the recurrent state. The earlier record analyses support an insulin effect near −1.9 per unit-hour, but norepinephrine estimates remain sensitive to adjustment. I’ll recover the working helpers next so we can test a targeted correction without rebuilding them.

```python
import inspect
import pandas as pd

print("=== NOTEBOOK FUNCTIONS ===")
notebook_functions = []
for name, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = ""
        notebook_functions.append((name, sig))
for name, sig in notebook_functions:
    print(name + sig[:180])

print("\n=== DATA TABLES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns)[:14])

print("\n=== KEY VARIABLES ===")
keys = ("trans", "step", "graft", "rob", "rd_", "causal", "contrast",
        "cache", "target", "beta", "checkpoint", "best", "spec", "prior")
shown = 0
for name, obj in list(globals().items()):
    if name.startswith("_") or not any(k in name.lower() for k in keys):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        description = f"{type(obj).__name__} {tuple(obj.shape)}"
    elif isinstance(obj, dict):
        description = "dict keys=" + repr(list(obj.keys())[:8])
    elif isinstance(obj, (str, int, float, bool, tuple)):
        description = repr(obj)[:180]
    else:
        continue
    print(name, description)
    shown += 1
    if shown >= 65:
        print("(inventory limited to 65 entries)")
        break
```
