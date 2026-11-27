The single-drug calculation matched the model, but the full-stream reconstruction still has meaningful errors. I’ll recover the architecture and kernel variables before designing an edit; an exact reconstruction would support using local dose perturbations, while persistent errors would make them unreliable.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== KERNEL INVENTORY ===")
for key, value in list(globals().items()):
    if key.startswith("_") or callable(value):
        continue
    typ = type(value).__name__
    if hasattr(value, "shape"):
        detail = f"shape={value.shape}"
    elif isinstance(value, dict):
        detail = f"len={len(value)}, keys={list(value)[:12]}"
    elif isinstance(value, (list, tuple)):
        detail = f"len={len(value)}"
    elif isinstance(value, (str, int, float, bool)):
        detail = repr(value)[:120]
    else:
        continue
    print(f"{key}: {typ}, {detail}")
print("=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPython runs: 13; forwards: 3. Restart recovery: inspecting architecture and live variables. No weights edited; full-stream posterior reconstruction is not yet exact.\n")
```