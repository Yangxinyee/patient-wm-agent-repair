The last run stopped because 53 drug bundles did not match the policy-target table; no weights were changed. I’ll recover the architecture, earlier findings, and retained arrays before fixing that alignment. An identifiable indexing mismatch would support proceeding; missing or inconsistent source records would require revising the feature construction.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
_notes = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK ===")
print(_notes[:16000])
print("=== RECENT NOTEBOOK ===")
print(_notes[-5500:])
print("=== RETAINED DATA / FEATURES ===")
_snapshot = list(globals().items())
for _name, _obj in _snapshot:
    if _name.startswith("_"):
        continue
    if isinstance(_obj, pd.DataFrame):
        print(_name, "DataFrame", _obj.shape, list(_obj.columns))
    elif isinstance(_obj, np.ndarray) and _obj.size >= 100:
        print(_name, "array", _obj.shape, _obj.dtype)
    elif hasattr(_obj, "shape") and type(_obj).__module__.startswith("torch") and _obj.numel() >= 100:
        print(_name, "tensor", tuple(_obj.shape))
    elif inspect.isfunction(_obj) and _obj.__module__ == "__main__":
        print(_name, "function", str(inspect.signature(_obj)))
    elif isinstance(_obj, dict) and any(s in _name.lower() for s in
                                         ("repair", "policy", "innovation", "latent", "panel", "match")):
        print(_name, "dict", len(_obj), list(_obj.keys())[:20])
print("=== FORWARD ARTIFACTS ===")
for _d in sorted(Path("/work").glob("fwd_*")):
    print(_d.name, [(p.name, p.stat().st_size) for p in _d.iterdir()])
print("=== CURRENT EDITS ===", edited())
```
