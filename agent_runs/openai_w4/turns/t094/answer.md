```python
from pathlib import Path
import numpy as np, pandas as pd, torch
print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nRETAINED OBJECTS")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape, list(v.columns)[:18])
    elif isinstance(v, (np.ndarray, torch.Tensor)) and getattr(v, "ndim", 0) > 0:
        if getattr(v, "size", 0) != 0:
            print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict) and len(v) > 1:
        print(k, "dict", len(v), list(v.keys())[:12])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function", getattr(getattr(v, "__code__", None), "co_varnames", ())[:14])
print("\nCURRENT EDITS", edited())
```
