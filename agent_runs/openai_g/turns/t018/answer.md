The edit made all four predicted rollout shifts exactly zero, with no factual-fit loss on the first 100 tuning stays. This establishes dose invariance, not causal truth. I’ll inspect the cached analysis before checking later-hour effects and the remaining stays. Consistently near-zero effects would support the edit; persistent dose effects or an NLL increase above 0.005 would challenge it.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:14000])

print("\n=== CACHED DATA AND FUNCTIONS ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, np.ndarray):
        print(name, "ndarray", obj.shape, obj.dtype)
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, "columns:", list(obj.columns))
    elif isinstance(obj, pd.Series):
        print(name, "Series", obj.shape, obj.dtype)
    elif inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            print(name, inspect.signature(obj))
        except (TypeError, ValueError):
            pass
    elif isinstance(obj, dict) and name not in {"SHAPES", "VOCAB"}:
        print(name, "dict keys:", list(obj)[:30])

print("\n=== CURRENT EDITS ===")
for name in edited():
    w, w0 = load(name), load_original(name)
    print(name, tuple(w.shape),
          "max absolute change:", float((w - w0).abs().max()) if hasattr(w, "abs")
          else float(np.max(np.abs(w - w0))))
```
