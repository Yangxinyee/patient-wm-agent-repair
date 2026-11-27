The last run failed because stay and hour are index levels, not columns; it did not change any weights. The current edit remains factual-safe on the tested subset. I’ll recover the dose units and retained calibration objects before making another edit. Agreement between the record-based estimates and rollout aggregation would support the targets; a unit mismatch would require correcting them.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

snapshot252 = list(globals().items())
print("\nRETAINED FUNCTIONS")
print(sorted(n for n, v in snapshot252 if inspect.isfunction(v)
             and not n.startswith("_")))

terms252 = ("panel", "kernel", "target", "projection", "irf", "dose",
            "candidate", "cand", "repair", "channel", "backup", "screen",
            "passing", "policy", "innovation", "instrument")
print("\nRELEVANT OBJECTS")
for n, v in snapshot252:
    if n.startswith("_") or inspect.ismodule(v) or inspect.isfunction(v):
        continue
    if any(t in n.lower() for t in terms252):
        if isinstance(v, dict):
            print(n, "dict keys:", list(v.keys())[:24])
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, "length", len(v))
        elif hasattr(v, "shape"):
            print(n, type(v).__name__, "shape", v.shape)
        elif isinstance(v, (str, int, float, bool)):
            print(n, repr(v)[:180])
        else:
            print(n, type(v).__name__)

print("\nPANEL INDEX AND DOSE UNITS")
seen252 = set()
for n, v in snapshot252:
    if (isinstance(v, pd.DataFrame)
        and {"insulin", "norepinephrine"}.issubset(v.columns)
        and id(v) not in seen252):
        seen252.add(id(v))
        print(n, "shape", v.shape, "index names", v.index.names)
        print(v[["insulin", "norepinephrine"]].head(4).to_string())
        print(v[["insulin", "norepinephrine"]]
              .describe(percentiles=[.5, .9, .99]).round(5).to_string())
        if len(seen252) >= 3:
            break

print("\nRAW RECORD SAMPLE")
print(pd.read_csv("/data/train.csv", nrows=8).to_string(index=False))
print("\nCURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 252\n"
            "Run 251 failed on MultiIndex column selection, not on estimation. "
            "Recovering architecture, retained interfaces, and raw/panel dose units. "
            "No weights changed; gain-1.25/gate-28 remains the saved candidate.\n")
```