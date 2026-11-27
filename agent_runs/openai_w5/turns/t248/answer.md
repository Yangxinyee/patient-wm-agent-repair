The last result shows that the local reconstruction does not reproduce the GPU’s factual loss, so its calibration estimates cannot yet be trusted. I’ll inspect the architecture and recover the saved objects before changing weights. Matching individual GPU predictions would validate the reconstruction; continued disagreement would refute it.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== PERSISTENT OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        keys = list(obj)
        print(name, "dict", len(keys), repr(keys[:16]))
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj[:4])[:180])
    elif inspect.isfunction(obj):
        try:
            print(name, "function", str(inspect.signature(obj)))
        except Exception:
            print(name, "function")
    elif isinstance(obj, (str, int, float, np.integer, np.floating)):
        print(name, repr(obj)[:180])

notes = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION AND RECONSTRUCTION NOTES ===")
print(notes[:14000])
print("\n=== WORK FILES ===")
print(sorted(str(p) for p in Path("/work").iterdir()))
```
