The last calculation exposed a substantial target discrepancy: raw-dose, row-weighted estimates are larger than the earlier cached estimates. I will verify the kernels’ units and identification evidence before changing weights. Agreement under the exact evaluation convention would support these targets; a units mismatch would refute that calibration.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

r239_globals = dict(globals())
print("=== RETAINED INTERFACES ===")
for name, obj in sorted(r239_globals.items()):
    if name.startswith("__"):
        continue
    if any(s in name.lower() for s in
           ("kernel", "target", "repair", "candidate", "build", "panel", "backup",
            "impulse", "calib", "channel", "replay", "factual")):
        if callable(obj):
            try:
                desc = str(inspect.signature(obj))
            except Exception:
                desc = "(signature unavailable)"
        elif isinstance(obj, dict):
            desc = "dict keys=" + repr(list(obj)[:18])
        elif hasattr(obj, "shape"):
            desc = f"{type(obj).__name__} shape={obj.shape}"
        else:
            desc = type(obj).__name__
        print(name, desc)

r239_notes = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION NOTES ===")
print(r239_notes[:14000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 239\n"
            "Recovering architecture, dose-unit definitions, identification evidence, "
            "and retained repair interfaces before accepting Run-238 raw-rate targets. "
            "No weights changed. Row-versus-stay aggregation and kernel units remain "
            "explicit calibration uncertainties.\n")
```