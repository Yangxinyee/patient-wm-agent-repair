The separator identifies insulin-active states almost perfectly and remains stable under dose changes. An activity gate may therefore preserve factual fit while allowing a stronger insulin correction, but this still needs rollout validation. I will recover the architecture and existing helpers before making a reversible trial.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes_text[:9500])
print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(s for s in notes_text.splitlines() if s.startswith("#")))

print("\n=== KERNEL FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and not name.startswith("__"):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\n=== KERNEL DATA / CHECKPOINT INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("__") or name in {"notes_text", "obj"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), "keys:", repr(keys[:14])[:450])
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(name, "table", obj.shape, list(obj.columns)[:18])
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        print(name, type(obj).__name__, len(obj), repr(obj[:5])[:220])

print("\nCURRENT EDITS:", edited())
```
