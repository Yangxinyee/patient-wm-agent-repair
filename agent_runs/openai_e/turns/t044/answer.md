```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14500])
print("\n=== PERSISTENT OBJECT INVENTORY ===")
for key, obj in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(key, type(obj).__name__, tuple(obj.shape),
              list(obj.columns)[:18] if isinstance(obj, pd.DataFrame) else "")
    elif isinstance(obj, dict):
        print(key, "dict", list(obj.keys())[:18])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function")
print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])
print("\n=== FORWARD OUTPUT FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 41: restart recovery\n"
            "Drug posterior is verified, but recurrent reconstruction remains inaccurate. "
            "Recovering architecture, causal evidence, and retained objects before an actual reversible edit. "
            "No weights edited. Python runs: 41; forward passes: 3.\n")
```
