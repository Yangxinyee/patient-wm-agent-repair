The rollout still misses the insulin estimates: **−19.55** for doubling and **+1.04** for halving. This refutes the idea that the local state correction was enough. I’ll inspect the rollout levels and repair implementation next, to distinguish saturation or drift from an incorrect dose-response mapping.

```python
import pandas as pd, numpy as np, re
from pathlib import Path

notes270 = Path("/work/notes.md").read_text().splitlines()
print("=== RECORD-DERIVED ROLLOUT REFERENCES ===")
for line in notes270:
    if "reference shifts" in line.lower() or "reference means" in line.lower():
        print(line[:1700])
print("=== KEY IMPLEMENTATION HISTORY ===")
for run in [193, 200, 216, 222, 267, 268]:
    found = [x for x in notes270 if re.search(rf"Python runs?:\s*{run}\b|Python run {run}\b", x)]
    for line in found[:2]:
        print(line[:1200])

roll270 = pd.read_csv("/work/fwd_22/dose_edit_edited.csv")
print("=== ROLLOUT COLUMNS AND SAMPLE ===")
print(roll270.columns.tolist())
print(roll270.head(4).to_string(index=False))
print(roll270.describe(include="all").to_string()[:2300])

print("=== RELEVANT GLOBAL NAMES ===")
selected270 = [n for n in globals() if re.search(
    r"267|222|249|268|260|262|264|target|reference|kernel|checkpoint|roll|repair|posterior|prelude",
    n, re.I)]
print(" ".join(selected270))
print("=== CURRENT CANDIDATE VARIABLES ===")
for n in [x for x in selected270 if "267" in x or x in ("checkpoint249", "checkpoint222")]:
    o = globals()[n]
    if isinstance(o, dict):
        print(n, {str(k): (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                  for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        print(n, "length", len(o), str(o)[:500])
    elif hasattr(o, "shape"):
        print(n, tuple(o.shape))
    else:
        print(n, str(o)[:400])

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 270; forwards 22. Candidate gamma=.1 factual delta +.00096 but actual rollouts (-19.551,+1.037,+1.719,-1.206) remain asymmetrically wrong for insulin. Local total-state constraints did not solve rollout behavior. Inspecting rollout levels, exact targets and retained implementation; no new edits.\n")
```
