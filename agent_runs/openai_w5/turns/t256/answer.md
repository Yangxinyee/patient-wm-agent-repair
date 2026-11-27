The architecture distinguishes posterior states used for updates from prior samples used for predictions. That distinction may explain the mismatch, but it is not yet established. I’ll recover the existing helpers and record-derived effect estimates so we can test it without changing weights.

```python
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== EARLY NOTES ===")
print(Path("/work/notes.md").read_text()[:5600])

print("\n=== LOCAL FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and hasattr(obj, "__code__") and name not in {"load", "load_original", "save_edit", "drop_edit", "edited"}:
        try:
            print(name, str(inspect.signature(obj)))
        except Exception:
            print(name)

print("\n=== TABLES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame) and not name.startswith("__"):
        print(name, obj.shape, list(obj.columns)[:16])

print("\n=== CALIBRATION OBJECT NAMES ===")
for name, obj in sorted(list(globals().items())):
    if any(s in name.lower() for s in ("layout", "control", "strength", "sensor", "center", "causal", "head", "predict", "state", "target", "factual", "effect")):
        if isinstance(obj, dict):
            print(name, "dict", repr(list(obj.keys())[:12]))
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (int, float, str, list, tuple)):
            print(name, repr(obj)[:400])

print("\n=== FORWARD FILES ===")
for folder in ["/work/fwd_15", "/work/fwd_16", "/work/fwd_17", "/work/fwd_19", "/work/fwd_20"]:
    p = Path(folder)
    if p.exists():
        print(folder, [x.name for x in p.iterdir()])
```
