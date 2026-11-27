The new states reflect the longer-lived insulin channel, but do not show whether it can meet the factual-fit limit. I will recover the calibration machinery and record-derived targets before recalibrating. The trial is worth retaining only if factual loss falls below the +0.005 limit and dose-response accuracy improves.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

_notes271 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND EDIT DESIGN ===")
print(_notes271[:20000])

print("\n=== KERNEL OBJECTS ===")
for _k271 in sorted(list(globals())):
    if _k271.startswith("__"):
        continue
    _v271 = globals()[_k271]
    if callable(_v271):
        print(_k271, "callable")
    elif isinstance(_v271, dict):
        print(_k271, "dict", len(_v271), "keys:", list(_v271)[:12])
    elif hasattr(_v271, "shape"):
        print(_k271, type(_v271).__name__, _v271.shape)
    elif isinstance(_v271, (list, tuple)):
        print(_k271, type(_v271).__name__, len(_v271))
    elif isinstance(_v271, (str, int, float, bool)):
        print(_k271, repr(_v271)[:160])

_s271 = np.load("/work/fwd_27/states_edited.npz")
print("\n=== FRESH STATES ===")
print({k: (v.shape, str(v.dtype)) for k, v in _s271.items()})
print("stay IDs:", np.unique(_s271["stay_id"])[:12])
print("currently edited:", edited())
```
