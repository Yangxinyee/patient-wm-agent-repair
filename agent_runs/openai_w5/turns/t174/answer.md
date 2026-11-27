The edit now gives the correct directions, but halving responses are too weak relative to the record-derived estimates. Doubling insulin is also underpowered. I’ll inspect the installed calibration and hourly shifts before increasing strength: asymmetric clipping would call for a different correction, not merely a larger one.

```python
from pathlib import Path
import re
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

note = """
## Forward 16–17: sensor-specific candidate GPU validation
- Tuning A factual value NLL: original 1.63933, edited 1.63946, delta +0.00013.
- Paired finite rollouts: insulin x2 -5.591, half +0.901; NE x2 +4.083, half -1.052.
- Record-derived tuning-A estimates: -11.893, +5.9465, +4.4973, -2.2487.
- Correct directions, but strong doubling/halving asymmetry. Increasing strengths alone is not yet justified.
- Investigating exposure transport, centering, nonlinear head response, and hourly feedback.
"""
with Path("/work/notes.md").open("a") as f:
    f.write(note)

print("=== EARLY NOTEBOOK: IDENTIFICATION AND ORIGINAL RESPONSE ===")
print(Path("/work/notes.md").read_text()[:18000])

print("=== PERSISTENT GLOBAL NAMES ===")
snapshot150 = list(globals().items())
print(", ".join(k for k, v in snapshot150 if not k.startswith("_")))

print("=== CALIBRATION / HEAD / CHANNEL OBJECTS ===")
for k, v in snapshot150:
    if k.startswith("_") or not re.search(
        r"head|tilt|caus|target|effect|proxy|clean|pair|strength|transport|147|148|149|rho|beta",
        k, re.I
    ):
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v.keys())[:45])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v), "preview", str(v[:8])[:700])
    elif isinstance(v, (str, int, float, bool)):
        print(k, repr(v)[:700])
    elif callable(v):
        print(k, "callable")
    else:
        print(k, type(v).__name__)

print("=== DOSE-ROLLOUT FILE LAYOUT ===")
for model150 in ["original", "edited"]:
    dose150 = pd.read_csv(f"/work/fwd_18/dose_edit_{model150}.csv")
    print(model150, dose150.shape, dose150.columns.tolist())
    print(dose150.head(8).to_string(index=False))
```
