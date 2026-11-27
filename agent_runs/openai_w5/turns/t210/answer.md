The refreshed encoding now matches the current weights; the stale cache was misleading. The last check stopped because a helper lacked its unit argument. I’ll recover the helper signatures and replay setup, then test transport again. Agreement with the edited GPU states would validate that test; disagreement would leave it unresolved.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())

print("=== RELEVANT HELPERS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and any(
        s in name.lower() for s in
        ("atom", "replay", "bundle", "encoder", "encode", "transport", "cpu", "step")
    ):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("=== RELEVANT OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or inspect.isfunction(obj):
        continue
    if any(s in name.lower() for s in
           ("cache", "s4", "replay", "transport", "effect", "causal", "sensor",
            "strength", "head", "dose", "world", "layout", "checkpoint")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:24])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        elif isinstance(obj, (str, int, float, bool, type(None))):
            print(name, repr(obj)[:180])

notes_text = Path("/work/notes.md").read_text()
print("=== EARLY RECORD-DERIVED EVIDENCE ===")
print(notes_text[:14000])
```
