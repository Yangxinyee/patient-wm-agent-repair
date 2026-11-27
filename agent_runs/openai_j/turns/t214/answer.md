```python
from pathlib import Path
import inspect, numpy as np, pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("\nNOTEBOOK: FIRST 14000 CHARACTERS")
print(notes_text[:14000])

print("\nPERSISTENT FUNCTIONS")
for key, obj in list(globals().items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(signature unavailable)"
        print(key, sig)

print("\nRELEVANT PERSISTENT VARIABLES")
for key, obj in list(globals().items()):
    if any(s in key.lower() for s in
           ["map", "cal", "raw", "front", "best", "combined", "compress", "install"]):
        if isinstance(obj, np.ndarray):
            print(key, "ndarray", obj.shape, obj.dtype)
        elif isinstance(obj, pd.DataFrame):
            print(key, "DataFrame", obj.shape, list(obj.columns))
        elif isinstance(obj, dict):
            print(key, "dict keys", list(obj)[:35])
        elif not inspect.isfunction(obj) and not inspect.ismodule(obj):
            print(key, type(obj).__name__, str(obj)[:250])

for filename in ["best_combined_v1.npz", "best_insulin_v1.npz"]:
    with np.load("/work/" + filename) as archive:
        print("\nARCHIVE", filename,
              [(k, archive[k].shape) for k in archive.files])
```
