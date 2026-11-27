The new attention channel reliably carries the insulin signal, but replacing the original head leaves a compensation error whose effect on factual predictions is unknown. I’ll inspect the candidate and transition code before testing it; a usable repair must change dose responses without exceeding the factual-loss budget.

```python
from pathlib import Path
import numpy as np

print(Path("/model/ARCHITECTURE.md").read_text())
print("\n--- retained candidate and diagnostic variables ---")
for name in sorted(globals()):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in (
        "cand", "repair", "comp", "att", "gate", "innov", "kernel",
        "causal", "panel", "prelude", "transition", "inject", "sens"
    )):
        obj = globals()[name]
        if callable(obj):
            print(name, "callable")
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:25])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        else:
            print(name, repr(obj)[:180])
print("\n--- notebook opening ---")
print(Path("/work/notes.md").read_text()[:16000])
print("\ncurrent edits:", edited())
```
