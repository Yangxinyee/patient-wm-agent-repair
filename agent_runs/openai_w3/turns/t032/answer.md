```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== TENSORS ===")
for name in NAMES:
    print(name, tuple(SHAPES[name]))
print("=== VOCAB TOP LEVEL ===")
for key, value in VOCAB.items():
    if isinstance(value, dict):
        print(key, "keys:", list(value))
    elif isinstance(value, list):
        print(key, "length:", len(value), "sample:", value[:8])
    else:
        print(key, value)
print("=== LIVE VARIABLES ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if callable(value):
        print(key, "callable")
    elif hasattr(value, "shape"):
        print(key, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict):
        print(key, "dict", len(value), "keys:", list(value)[:12])
    elif isinstance(value, (list, tuple)):
        print(key, type(value).__name__, len(value))
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 28; forwards 3. State layout and original weights verified. Recovering architecture and live helpers before selecting a weight-only edit. No edits.\n")
```