```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or not inspect.isfunction(obj):
        continue
    if getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\n=== RETAINED ARRAYS / TABLES / DICTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape, list(obj.columns)[:12])
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), [str(k) for k in keys[:8]])

print("\n=== EARLY NOTEBOOK: IDENTIFICATION EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:10000])
print("\n=== ACTIVE EDITS ===")
print(edited())
```
