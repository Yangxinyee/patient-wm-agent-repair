The calibrated model’s loss is 0.00436 nats above the original on these 100 stays—within the limit, but with little margin. This does not yet establish overall factual fit or dose-response accuracy. I’ll recover the earlier comparisons and edit details; consistent improvement across the remaining stays would support keeping this calibration.

```python
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes296 = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD ANALYSIS / EDIT DESIGN ===")
print(notes296[:14000])
print("\n=== NOTEBOOK SECTION INDEX ===")
print("\n".join(re.findall(r"^## .*", notes296, flags=re.M)[-70:]))

REC296 = {"factual": {}, "dose": {}, "groups": []}
dirs296 = sorted(
    [p for p in Path("/work").glob("fwd_*") if p.is_dir()],
    key=lambda p: int(p.name.split("_")[-1])
)
for p in dirs296:
    k = int(p.name.split("_")[-1])
    if k < 24:
        continue
    for model in ("original", "edited"):
        f = p / f"factual_{model}.csv"
        if f.exists():
            d = pd.read_csv(f)
            REC296["factual"][(k, model)] = d
            nc = [c for c in d.columns if "nll" in c.lower()]
            sc = next((c for c in d.columns if "stay" in c.lower()), None)
            ids = sorted(d[sc].unique().tolist()) if sc else []
            print("FACTUAL", k, model, "rows", len(d),
                  "stays", len(ids),
                  "losses", {c: float(d[c].mean()) for c in nc})
    for model in ("original", "edited"):
        f = p / f"dose_edit_{model}.csv"
        if f.exists():
            d = pd.read_csv(f)
            REC296["dose"][(k, model)] = d
            print("DOSE", k, model, "columns", list(d.columns))
            print(d.head(4).to_string(index=False))

seen296 = set()
for (k, model), d in sorted(REC296["factual"].items(), reverse=True):
    if model != "original":
        continue
    sc = next((c for c in d.columns if "stay" in c.lower()), None)
    if sc is None:
        continue
    ids = sorted(d[sc].unique().tolist())
    key = tuple(ids)
    if len(ids) == 100 and key not in seen296:
        seen296.add(key)
        REC296["groups"].append({"forward": k, "stay_column": sc, "ids": ids})
        if len(REC296["groups"]) == 4:
            break

print("\n=== FOUR MOST RECENT DISTINCT 100-STAY GROUPS ===")
print(json.dumps(REC296["groups"], default=str))
latest296 = REC296["factual"].get((41, "edited"))
if latest296 is not None:
    print("\n=== FACTUAL SCHEMA ===")
    print(list(latest296.columns))
    print(latest296.head(3).to_string(index=False))

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
for name in sorted(list(globals())):
    if not re.search(r"(BASE|base|center|control|signal|target|truth|coef|beta|tau|295|289|285|900|1250)", name):
        continue
    obj = globals()[name]
    if isinstance(obj, (int, float, str, bool, type(None))):
        print(name, repr(obj)[:250])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys", list(obj)[:20])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj[:8])[:300])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 296: restart recovery and calibration audit\n"
            "- Latest calibrated factual group: 1.62788 vs 1.62352, delta +0.00436.\n"
            "- Recovered architecture, earlier record analysis, paired factual files, "
            "rollout files, and validation-group IDs. No weights changed.\n"
            "- Calibration remains provisional until pooled factual validation and "
            "paired finite-rollout checks; training-record causal estimates are not "
            "observed intervention truth.\n")
```
