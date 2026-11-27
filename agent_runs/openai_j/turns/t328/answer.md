The current edit passes the factual-loss limit on 400 tuning stays, but the uncertainty interval crosses that limit. I’ll recover the records-derived effect estimates and candidate builders to find a safer version. My hypothesis is that calibration can preserve the dose responses while reducing factual loss; held-out loss and paired rollouts must both support it.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes282 = Path("/work/notes.md").read_text()
print("=== EARLY CAUSAL NOTEBOOK ===")
print(notes282[:14000])

print("=== RECORD-DERIVED ESTIMATES ===")
for filename282 in ["kernel_targets.csv", "dynamic_iv.csv", "sensor_iv.csv", "dml_results.csv"]:
    table282 = pd.read_csv("/work/" + filename282)
    print(filename282, "shape", table282.shape)
    print(table282.head(24).to_string(index=False))

print("=== CALIBRATION FRONTIER ===")
front282 = pd.read_csv("/work/calibration_strength_frontier265.csv")
print(front282.sort_values("tune_dNLL").head(14).to_string(index=False))
print(front282[(front282.insulin == 0.90) & (front282.MAP_gain >= 7)].to_string(index=False))

print("=== CANDIDATE CONSTRUCTION OBJECTS ===")
for name282, obj282 in list(globals().items()):
    if any(tag282 in name282 for tag282 in ["265", "268", "242", "247"]):
        if callable(obj282):
            try:
                print(name282, inspect.signature(obj282))
            except (ValueError, TypeError):
                print(name282, type(obj282).__name__)
        elif isinstance(obj282, dict):
            print(name282, "dict keys", list(obj282.keys())[:18])
        elif hasattr(obj282, "shape"):
            print(name282, type(obj282).__name__, tuple(obj282.shape))
        elif isinstance(obj282, (list, tuple)):
            print(name282, type(obj282).__name__, "length", len(obj282))
        elif isinstance(obj282, (str, int, float)):
            print(name282, repr(obj282)[:180])

with open("/work/notes.md", "a") as file282:
    file282.write(
        "\n## Run 282: safety-first recovery\n"
        "- Current calibrated four-tensor edit remains installed; no weights changed.\n"
        "- Recovering architecture, records-derived causal estimates, and calibration builders.\n"
        "- Observed pooled tuning dNLL +0.00431938 passes, but bootstrap safety margin is weak.\n"
        "- Counts: 282 Python runs, 38 forward passes.\n"
    )
```
