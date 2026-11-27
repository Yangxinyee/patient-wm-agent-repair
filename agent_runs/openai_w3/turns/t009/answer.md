```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])
print("\n=== VOCAB SUMMARY ===")
for key, value in VOCAB.items():
    text = repr(value)
    print(key, text if len(text) < 4500 else text[:4500] + " ...")
print("\n=== KERNEL VARIABLES ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(value, (np.ndarray, torch.Tensor)):
        print(key, type(value).__name__, tuple(value.shape), str(value.dtype))
    elif isinstance(value, dict):
        print(key, "dict", list(value.keys())[:30])
    elif isinstance(value, (list, tuple)):
        print(key, type(value).__name__, "length", len(value))
    elif callable(value):
        if getattr(value, "__module__", "") == "__main__":
            print(key, "function")
print("\n=== FORWARD FILES ===")
for folder in ["/work/fwd_1", "/work/fwd_2"]:
    for path in Path(folder).iterdir():
        print(path, path.stat().st_size)
print("\n=== CURRENT EDITS ===")
print(edited())
```
