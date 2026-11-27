```python
from pathlib import Path
import numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== NOTEBOOK: EARLIER CAUSAL EVIDENCE ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:18000])
print("\n=== KERNEL OBJECT INVENTORY ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"notes_text", "obj"}:
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns)[:24])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        if obj.size if isinstance(obj, np.ndarray) else obj.numel():
            print(name, type(obj).__name__, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, dict) and (
        any(s in name.lower() for s in ["contrast", "head", "state", "fit", "meta", "coef", "mc", "cache"])
        or any(k in obj for k in ["insulin", "norepinephrine", ("insulin", 2.0)])
    ):
        print(name, "dict", list(obj.keys())[:25])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        import inspect
        try:
            print(name, "function", inspect.signature(obj))
        except Exception:
            print(name, "function")
print("\nActive edits:", edited())
```
