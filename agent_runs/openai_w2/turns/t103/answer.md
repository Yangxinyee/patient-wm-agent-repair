```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:10000])
print("\n=== KERNEL OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith("_") or name in {"In", "Out"}:
        continue
    obj = globals()[name]
    if callable(obj):
        if getattr(obj, "__module__", "") == "__main__":
            print(name, "function")
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (dict, list, tuple)):
        print(name, type(obj).__name__, "length", len(obj))
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape)
print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in ("empty", "embed", "enc", "post", "prior", "transition")):
        print(name, SHAPES[name])
print("Current edits:", edited())
```
