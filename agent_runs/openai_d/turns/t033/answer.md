The reconstructed reading head is now close enough that the remaining error could be sampling noise. That makes a targeted head correction plausible, but not yet validated. I’ll recover the architecture and cached analyses before choosing an edit; a useful correction must track dose innovations while preserving factual predictions.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTES ===")
print(Path("/work/notes.md").read_text()[:16000])
print("\n=== RELEVANT KERNEL VARIABLES ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            print("FUNCTION", name, inspect.signature(obj))
        except Exception:
            print("FUNCTION", name)
    elif any(k in name.lower() for k in
             ["state", "head", "feat", "prior", "kernel", "causal", "resid",
              "align", "config", "factual", "roll", "result", "target"]):
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:35])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj), repr(obj)[:500])

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])
print("\n=== SAVED FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
print("\nCurrent edits:", edited())
```
