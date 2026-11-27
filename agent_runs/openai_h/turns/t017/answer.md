The latest factual test worsened NLL by only 0.00040 nats, well below the 0.005 limit. That supports retaining the edit provisionally. Next I’ll recover the earlier rollout results and kernel context: zero dose shifts would confirm invariance; nonzero shifts would reveal a remaining dose-dependent route.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== EXISTING KERNEL OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict) and name not in {"__builtins__"}:
        print(name, "dict", len(obj), "keys:", list(obj)[:12])

print("\n=== SAVED FORWARD RESULTS ===")
for folder in sorted(Path("/work").glob("fwd_*"),
                     key=lambda p: int(p.name.split("_")[-1])):
    print("\n", folder.name, sorted(p.name for p in folder.iterdir()))
    for path in sorted(folder.glob("*.csv")):
        table = pd.read_csv(path)
        print(path.name, "shape:", table.shape, "columns:", list(table.columns))
        print(table.head(4).to_string(index=False))
        if "dose_edit" in path.name:
            groups = [c for c in table.columns
                      if c in {"edit", "intervention", "dose_edit", "scenario"}]
            numeric = table.select_dtypes(include=np.number).columns.tolist()
            if groups:
                print(table.groupby(groups, dropna=False)[numeric].mean().to_string())
            else:
                print(table[numeric].agg(["mean", "min", "max"]).to_string())
        elif "factual" in path.name:
            nll_cols = [c for c in table.columns if "nll" in c.lower()]
            print("Mean NLL columns:", table[nll_cols].mean().to_dict())

print("\n=== TUNING STAY IDS ===")
tuning_audit = pd.read_csv("/data/tuning.csv")
print("Columns:", list(tuning_audit.columns))
stay_col_audit = next(c for c in tuning_audit.columns
                      if c.lower() in {"stay", "stay_id", "icustay_id"})
tuning_ids_audit = sorted(tuning_audit[stay_col_audit].unique().tolist())
print("Number of stays:", len(tuning_ids_audit))
print("Stay IDs:", tuning_ids_audit)

print("\n=== CURRENT EDITS ===")
print(edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Restart audit: Python run 11, forward passes 5\n"
        "- Forward 5: original factual NLL 1.47456, edited 1.47496 "
        "on 100 tuning stays / 4,837 values; delta +0.00040, below +0.005.\n"
        "- Recovering earlier forward results and kernel objects before "
        "extending validation. Null drug effects remain a hypothesis "
        "supported by adjusted training-record estimates, not medical assumptions.\n"
    )
```