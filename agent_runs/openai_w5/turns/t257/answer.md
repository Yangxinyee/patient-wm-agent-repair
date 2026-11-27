The earlier held-out record analyses found consistent drug-specific associations after history adjustment, though their causal interpretation still depends on identification assumptions. The helper listing was truncated by large default arguments. I’ll inspect only compact signatures and relevant objects; no weights will change.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=== COMPACT FUNCTION SIGNATURES ===")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and hasattr(obj, "__code__"):
        c = obj.__code__
        print(name, list(c.co_varnames[:c.co_argcount + c.co_kwonlyargcount]))

print("\n=== TABLES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame) and not name.startswith("__"):
        print(name, obj.shape, list(obj.columns)[:12])

print("\n=== RELEVANT OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if any(s in name.lower() for s in ("layout", "control", "strength", "sensor", "center", "causal", "head", "predict", "state", "target", "effect")):
        if isinstance(obj, dict):
            print(name, "dict", repr(list(obj.keys())[:12]))
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (int, float, str)):
            print(name, repr(obj)[:180])

print("\n=== FORWARD FILES ===")
for folder in ["/work/fwd_15", "/work/fwd_16", "/work/fwd_17", "/work/fwd_19", "/work/fwd_20"]:
    p = Path(folder)
    if p.exists():
        print(folder, [x.name for x in p.iterdir()])

print("\n=== RECONSTRUCTION NOTES ===")
text = Path("/work/notes.md").read_text()
for section in text.split("\n## "):
    if any(s in section.split("\n")[0].lower() for s in ("20", "21", "22", "23", "24", "25", "147", "148", "149", "218", "219")):
        print(section[:900])
```
