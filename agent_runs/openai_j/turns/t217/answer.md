No weights changed. The inspection was truncated again because function signatures included large default arrays. I’ll print only compact parameter names and recent variable shapes. That should reveal how to install the calibrated candidate; until then, the validated combined edit remains active.

```python
import inspect, numpy as np, pandas as pd
from pathlib import Path

unique_funcs = {}
for k, v in list(globals().items()):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        unique_funcs[id(v)] = (k, v)
print("RECENT UNIQUE FUNCTIONS")
for k, v in list(unique_funcs.values())[-35:]:
    c = v.__code__
    print(k, c.co_varnames[:c.co_argcount + c.co_kwonlyargcount])

print("\nRECENT DATA VARIABLES")
compact_items = [(k, v) for k, v in list(globals().items())
                 if isinstance(v, (np.ndarray, pd.DataFrame, dict, list, tuple))
                 and not k.startswith("_")
                 and k not in ["func_items", "data_items", "unique_funcs", "compact_items"]]
for k, v in compact_items[-75:]:
    if isinstance(v, np.ndarray):
        print(k, "array", v.shape)
    elif isinstance(v, pd.DataFrame):
        print(k, "frame", v.shape, list(v.columns)[:8])
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:8])
    else:
        print(k, type(v).__name__, len(v))

for fn in ["best_combined_v1.npz", "best_insulin_v1.npz"]:
    with np.load("/work/" + fn) as z:
        print("ARCHIVE", fn, [(k, z[k].shape) for k in z.files])

nt = Path("/work/notes.md").read_text()
p = nt.find("## Causal identification checks")
print("\nIDENTIFICATION NOTES\n" + nt[p:p+1400])
```
