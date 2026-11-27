```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RECENT CANDIDATES AND HELPERS ===")
snapshot = list(globals().items())
for name, obj in snapshot:
    if name.startswith("_"):
        continue
    if any(tag in name.lower() for tag in
           ("154", "155", "156", "147", "145", "replay", "core_step",
            "means105", "post", "carrier", "transport")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", {
                str(k): (tuple(v.shape) if hasattr(v, "shape")
                         else type(v).__name__)
                for k, v in obj.items()
            })
        elif callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                print(name, "callable")
        else:
            text = repr(obj)
            print(name, text[:240])

print("\n=== ALL FUNCTION SIGNATURES ===")
for name, obj in snapshot:
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

print("\n=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 157; forwards 10. Posterior replay is authoritative "
            "(RMS 2.4e-5), but recurrent replay and readout calibration are "
            "not validated. Recovering architecture and retained transport/"
            "posterior candidates before further edits. Existing five-tensor "
            "zero-payload transport remains unchanged.\n")
```