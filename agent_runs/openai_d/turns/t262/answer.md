```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:18000])
print("=== EDITED TENSORS ===")
for name in edited():
    print(name, SHAPES[name])
print("=== LIVE VARIABLE NAMES ===")
print(" ".join(sorted(k for k in globals() if not k.startswith("__"))))
print("=== RECENT VARIABLE TYPES/SHAPES ===")
for k in sorted(list(globals())):
    if k.startswith(("c18", "c19", "c20", "c21", "c22", "r18", "r19", "r20", "r21", "r22")):
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:16] if isinstance(v, dict) else "")
```
