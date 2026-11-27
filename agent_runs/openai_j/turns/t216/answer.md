The architecture confirms that the current edit is confined to the value head, avoiding direct changes to the other prediction heads. The record analyses also warn against treating dose–reading associations as causal effects. The output was truncated before the candidate definitions, so I’ll retrieve those in a smaller report. I need to preserve the validated insulin edit and identify a reproducible MAP candidate before testing it.

```python
import inspect, numpy as np, pandas as pd
from pathlib import Path

print("FUNCTIONS")
func_items = [(k, v) for k, v in list(globals().items())
              if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__"]
for k, v in func_items:
    try:
        print(k, inspect.signature(v))
    except Exception:
        print(k)

print("\nLAST 65 DATA VARIABLES")
data_items = [(k, v) for k, v in list(globals().items())
              if isinstance(v, (np.ndarray, pd.DataFrame, dict, list, tuple))
              and k not in ["__builtins__", "func_items", "data_items"]]
for k, v in data_items[-65:]:
    if isinstance(v, np.ndarray):
        detail = f"array {v.shape} {v.dtype}"
    elif isinstance(v, pd.DataFrame):
        detail = f"frame {v.shape} columns={list(v.columns)[:12]}"
    elif isinstance(v, dict):
        detail = f"dict keys={list(v)[:12]}"
    else:
        detail = f"{type(v).__name__} length={len(v)}"
    print(k, detail)

for fn in ["best_combined_v1.npz", "best_insulin_v1.npz"]:
    with np.load("/work/" + fn) as z:
        print("\nARCHIVE", fn, [(k, z[k].shape) for k in z.files])

nt = Path("/work/notes.md").read_text()
p = nt.find("## Causal identification checks")
print("\nIDENTIFICATION NOTES")
print(nt[p:p+2500])
```
