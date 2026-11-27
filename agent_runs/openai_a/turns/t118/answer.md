```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_g = dict(globals())
print("\n=== NOTEBOOK-DEFINED HELPERS ===")
print(sorted(k for k, v in _g.items()
             if callable(v) and getattr(v, "__module__", None) == "__main__"
             and not k.startswith("_")))

print("\n=== RETAINED TABLES ===")
for k, v in sorted(_g.items()):
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, list(v.columns)[:24])

print("\n=== RECONSTRUCTION CONFIGURATION ===")
for k, v in sorted(_g.items()):
    if any(s in k.lower() for s in ("spec", "prefix", "recon", "candidate", "best", "unit")):
        if isinstance(v, (str, int, float, bool, tuple)):
            print(k, repr(v)[:700])
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:20])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))

print("\n=== EARLY IDENTIFICATION NOTES ===")
_notes = Path("/work/notes.md").read_text()
print(_notes[:16000])
```
