The action-only reconstruction is accurate, but END and measurement bundles still disagree with the model. I won’t base an edit on that incomplete reconstruction. Next I’ll recover the architecture and causal evidence; these should show whether the mismatch is a bookkeeping issue and which drug effects the records actually support.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:18000])
print("\n=== KERNEL INVENTORY ===")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"In", "Out", "notes_text"}:
        continue
    obj = globals()[key]
    if isinstance(obj, np.ndarray):
        print(key, "array", obj.shape, obj.dtype)
    elif hasattr(obj, "shape") and not callable(obj):
        print(key, type(obj).__name__, obj.shape)
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function")
    elif isinstance(obj, dict):
        print(key, "dict", len(obj), "keys:", list(obj)[:12])
print("\n=== CURRENT EDITS ===", edited())
```
