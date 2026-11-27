The probe found a possible norepinephrine response direction, but its factual variation makes safety uncertain. The active-MAP gate looks promising. I’ll recover the architecture and earlier evidence before constructing an edit; a useful candidate must improve dose responses without exceeding the factual-NLL budget.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
print("=== EARLIER NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14500])

snapshot133 = list(globals().items())
print("=== RECENT VARIABLES ===")
for k, v in snapshot133:
    if k.startswith(("r12", "r13", "r115", "r98", "r99", "r103", "r104", "r106")):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:30])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "len", len(v))
        elif callable(v):
            print(k, "callable", str(inspect.signature(v)))
        else:
            print(k, type(v).__name__, repr(v)[:160])
print("=== AVAILABLE HELPER FUNCTIONS ===")
for k, v in snapshot133:
    if callable(v) and getattr(v, "__module__", None) == "__main__":
        try:
            print(k, inspect.signature(v))
        except (TypeError, ValueError):
            print(k)
print("=== CURRENT EDITS ===", edited())
```
