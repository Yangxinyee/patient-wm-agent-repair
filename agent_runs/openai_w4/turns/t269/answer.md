The gate change reduced insulin’s half-dose shift from about +12.6 to +6.7, while retaining a substantial doubling response. That supports a gating contribution to the asymmetry, but norepinephrine remains poorly balanced. I’ll recover the record-derived kernels and edit construction next; matched, identically weighted targets will determine whether further calibration is justified.

```python
from pathlib import Path
import inspect, re
import numpy as np
import pandas as pd

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 234; forward passes used 22/50
Threshold-26 insulin gate candidate passed GPU factual: original 1.59352,
edited 1.59741, delta .00389 on 98 tuning stays. GPU reached-row rollout
shifts: insulin x2 -8.773, half +6.696; NE x2 +.572, half -3.204.
Relative to the previous candidate (-10.385, +12.633, +.730, -3.130),
the gate shift markedly reduced excessive insulin half-dose response.
This supports a computational gating contribution, not nonlinear ICU
pharmacology. Recovering matched record-derived targets and construction
before further calibration. No weights changed in Run 234.
""")

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

g234 = dict(globals())
print("=== RETAINED FUNCTIONS ===")
for name, value in sorted(g234.items()):
    if inspect.isfunction(value) and value.__module__ == "__main__":
        try:
            print(name, inspect.signature(value))
        except Exception:
            print(name)

print("=== RELEVANT RETAINED OBJECTS ===")
pattern234 = re.compile(
    r"kernel|target|caus|ridge|repair|connection|backup|channel|innovation|gate|impulse|effect"
)
for name, value in sorted(g234.items()):
    if pattern234.search(name.lower()) and not inspect.isfunction(value):
        desc = type(value).__name__
        if hasattr(value, "shape"):
            desc += " shape=" + str(tuple(value.shape))
        elif isinstance(value, dict):
            desc += " keys=" + repr(list(value.keys())[:16])
        elif isinstance(value, (list, tuple)):
            desc += " len=" + str(len(value))
        elif isinstance(value, (str, int, float)):
            desc += " value=" + repr(value)[:180]
        print(name, desc)

print("=== SAME-SUBSET ROLLOUT AGGREGATION ===")
for folder in ("/work/fwd_21", "/work/fwd_23"):
    p = Path(folder) / "dose_edit_edited.csv"
    if p.exists():
        t = pd.read_csv(p)
        t = t[t["reached"].astype(bool) & t["hour"].between(13, 24)]
        print(folder, "rows", t.groupby("edit")["shift"].mean().round(4).to_dict(),
              "equal-stay", t.groupby(["edit", "stay_id"])["shift"].mean()
                  .groupby("edit").mean().round(4).to_dict())
```
