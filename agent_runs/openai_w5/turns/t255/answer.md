The exact weight cache did not fix the local/GPU discrepancy, so the local head calculations are not yet trustworthy. I’ll check the architecture and prediction-input reconstruction before making another edit. Agreement on individual readings—not just average loss—would validate that reconstruction.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== KERNEL OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("__"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and name not in {"__builtins__"}:
        keys = list(obj.keys())
        print(name, "dict", len(keys), "keys", repr(keys[:10]))
    elif callable(obj) and getattr(obj, "__module__", "") in {"__main__", None}:
        print(name, "callable")

print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND HEAD DESIGN ===")
notes = Path("/work/notes.md").read_text()
print(notes[:18000])
```
