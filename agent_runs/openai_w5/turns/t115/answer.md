The available helpers cover both action encoding and rollout readouts. Before editing, I’ll recover the original rollout effects and the record-based effect estimates. A gain adjustment is worth testing only if the original response has the right sign; otherwise it could amplify the wrong behavior.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== EARLY IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:3200])

print("=== SAVED EFFECT ESTIMATES ===")
for name in ["dyn_estimates", "dynamic_alternatives", "est", "best_head_spec", "best_layout_spec"]:
    if name in globals():
        print(name, repr(globals()[name])[:1600])

print("=== FORWARD ROLLOUT FILES ===")
for path in sorted(Path("/work").glob("fwd_*/dose_edit_*.csv")):
    tab102 = pd.read_csv(path)
    print(str(path), tab102.shape, list(tab102.columns))
    grouping102 = [
        c for c in ["edit", "intervention", "drug", "multiplier", "readout"]
        if c in tab102.columns
    ]
    numeric102 = [
        c for c in tab102.select_dtypes(include=[np.number]).columns
        if c not in ["stay_id", "hour", "multiplier", "n", "n_paths"]
    ]
    if grouping102 and numeric102:
        print(tab102.groupby(grouping102, dropna=False)[numeric102].mean().to_string()[:1000])
    else:
        print(tab102.head(2).to_string(index=False)[:1000])
        print("numeric means", tab102[numeric102].mean().to_dict())

print("=== REMAINING OBJECT NAMES ===")
names102 = []
for name, obj in sorted(list(globals().items())):
    if name < "g" or name.startswith("_") or callable(obj):
        continue
    if isinstance(obj, (dict, list, tuple, np.ndarray, pd.DataFrame)):
        names102.append(name)
print(" ".join(names102)[:1800])
print("EDITS", edited())
```