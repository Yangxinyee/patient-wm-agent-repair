The fixed-history response is nearly symmetric. That points toward dose transport or feedback during rollout as the source of the asymmetry. I’ll inspect those mechanisms next; asymmetric rollout inputs would support this hypothesis, while symmetric inputs would refute it.

```python
import inspect, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== RECENT PERSISTENT OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith(("_r148", "_r149", "_r150", "_r151", "_r152", "_r153", "_r154", "_r155")):
        obj = globals()[name]
        if callable(obj):
            try:
                text = inspect.getsource(obj)
            except Exception:
                text = str(inspect.signature(obj)) if inspect.isfunction(obj) else str(type(obj))
            print(name, text[:6500])
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:35])
        elif isinstance(obj, pd.DataFrame):
            print(name, "DataFrame", obj.shape, list(obj.columns))
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (str, int, float, list, tuple)):
            print(name, repr(obj)[:600])
print("=== TRANSPORT/CACHE HELPERS ===")
for name in sorted(list(globals())):
    if any(k in name.lower() for k in ["reconstruct", "build_features", "head_features", "transport", "calibrat"]):
        obj = globals()[name]
        if callable(obj):
            try:
                print(name, inspect.getsource(obj)[:8000])
            except Exception:
                print(name, repr(obj))
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj.keys())[:30])
print("=== NOTEBOOK IDENTIFICATION SECTIONS ===")
notes156 = Path("/work/notes.md").read_text()
print(notes156[:18000])
```
