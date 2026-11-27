The reduced readout increases factual NLL by 0.00263, within the 0.005 limit on this sample. Its rollout effects remain untested. I’ll recover the architecture and live variables, then test whether it improves dose responses without exhausting that factual margin.

```python
from pathlib import Path
import re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

_g187 = dict(globals())
print("\n=== RECENT VARIABLES ===")
for name, obj in sorted(_g187.items()):
    if re.search(r"(167|173|174|175|176|177|178|179|180|185|186)$", name):
        if hasattr(obj, "shape"):
            description = f"{type(obj).__name__}, shape={tuple(obj.shape)}"
        elif isinstance(obj, dict):
            description = f"dict, keys={list(obj)[:16]}"
        elif callable(obj):
            description = "callable"
        else:
            description = f"{type(obj).__name__}: {str(obj)[:180]}"
        print(name, description)

print("\n=== STAY / AUDIT IDENTIFIERS ===")
for name, obj in sorted(_g187.items()):
    if re.search(r"(audit|tune.*ids|tuning.*ids|stay.*ids|ids.*tune)", name, re.I):
        if isinstance(obj, (list, tuple, np.ndarray)):
            print(name, "length=", len(obj), "first=", list(obj[:12]))
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))

print("\n=== IMPORTANT EXISTING OBJECTS ===")
for name in ["VOCAB", "kernels19", "readout179", "B167",
             "wire_out152", "R118", "P125"]:
    if name in _g187:
        obj = _g187[name]
        print(name, type(obj).__name__,
              tuple(obj.shape) if hasattr(obj, "shape")
              else list(obj)[:24] if isinstance(obj, dict)
              else str(obj)[:240])

print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 187 / restarted recovery\n"
        "Forward 11 (fwd_12): reduced readout at 0.70 strength gives "
        "factual NLL 1.63427 vs original 1.63164, delta +0.00263 "
        "on 100 tuning stays / 4751 readings. Within the limit on this "
        "sample; causal rollout benefit remains unvalidated. Recovering "
        "architecture, live interfaces and stay IDs before paired rollouts.\n"
    )
```