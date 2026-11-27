```python
from pathlib import Path
import inspect, json, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== REUSABLE FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if not name.startswith("_") and inspect.isfunction(obj):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

print("\n=== ANALYSIS OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        keys = list(obj)
        if len(keys) <= 30:
            print(name, "dict keys:", keys)
print("\nCurrent edits:", edited())
```
