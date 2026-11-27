The current edit amplifies dose-bin differences while preserving each drug’s average training embedding. That does not yet show better dose responses or preserved factual fit. I’ll recover the architecture and earlier evidence before testing it; stronger responses with NLL within the 0.005 limit would support this approach.

```python
from pathlib import Path
print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README\n", Path("/data/README.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("\nEARLIER NOTEBOOK\n", notes_text[:14000])
print("\nCURRENT EDITS:", edited())
print("\nKERNEL VARIABLES:")
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"In", "Out", "notes_text", "val"}:
        continue
    if hasattr(val, "shape"):
        print(key, type(val).__name__, tuple(val.shape))
    elif isinstance(val, dict):
        print(key, "dict", list(val.keys())[:12])
    elif isinstance(val, (list, tuple)) and len(val) < 200:
        print(key, type(val).__name__, "length", len(val))
```