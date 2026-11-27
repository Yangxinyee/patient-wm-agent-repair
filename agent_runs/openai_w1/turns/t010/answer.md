```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())
print("\n=== KERNEL VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, (pd.DataFrame, np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", list(v)[:12])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")
print("\n=== TENSOR SHAPES ===")
for n in NAMES:
    print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 7\nRe-reading architecture and inspecting persistent variables before a targeted algebraic edit. Reconstruction remains too inaccurate to use safely; no tensors edited.\nCounts: 7 Python runs, 3 forward passes.\n")
```
