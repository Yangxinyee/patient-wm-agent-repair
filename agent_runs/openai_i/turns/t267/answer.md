```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY EVIDENCE / VALIDATION NOTES ===")
notes239 = Path("/work/notes.md").read_text()
print(notes239[:18000])

print("\n=== RELEVANT KERNEL OBJECTS ===")
for name239 in sorted(list(globals())):
    if any(name239.endswith(str(k)) for k in [182, 209, 214, 222, 228, 230, 231, 233, 237, 238]):
        obj239 = globals()[name239]
        if callable(obj239):
            print(name239, "function")
        elif isinstance(obj239, dict):
            print(name239, "dict", list(obj239)[:16])
        elif hasattr(obj239, "shape"):
            print(name239, tuple(obj239.shape))
        elif isinstance(obj239, (tuple, list)):
            print(name239, type(obj239).__name__, len(obj239))
        elif isinstance(obj239, (str, float, int)):
            print(name239, str(obj239)[:160])

print("\n=== SAVED RESPONSE / ROUTE ARRAYS ===")
for file239 in ["response_directions231.npz", "small_route222.npz", "neutral209.npz"]:
    z239 = np.load("/work/" + file239)
    print(file239)
    for key239 in z239.files:
        a239 = z239[key239]
        if np.issubdtype(a239.dtype, np.number):
            print(key239, a239.shape, "norm", float(np.linalg.norm(a239)),
                  "max", float(np.max(np.abs(a239))) if a239.size else 0)

with open("/work/notes.md", "a") as f239:
    f239.write("\n## Python 239 — recover architecture and response scale\n"
               "- Neutral209 remains installed; rejected response233 is not active.\n"
               "- Testing whether large response-column norms amplify minute inactive leakage.\n"
               "- Recovering record evidence and replay contracts before a revised action-gated edit.\n")
```
