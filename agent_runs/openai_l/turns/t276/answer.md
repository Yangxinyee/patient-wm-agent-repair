The recovered fit statistics put the current candidate near +0.004 NLL on both offline audit sets, consistent with the latest GPU result. The output again cut off before the rollout comparison. I’ll retrieve only those summaries and stay IDs next; the key question is whether the stronger head materially improves the four dose readouts.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== CURRENT AND NEARBY FIT STATISTICS ===")
print("current:", result242)
for key, val in grid242.items():
    print(key, compact240(val))

print("=== RECENT ROLLOUTS ===")
for k in [13, 14, 15, 16, 17, 18]:
    folder = Path(f"/work/fwd_{k}")
    if not folder.exists():
        continue
    print(folder.name, [p.name for p in folder.iterdir()])
    for p in folder.glob("dose_edit*.csv"):
        table = pd.read_csv(p)
        print(p.name, "columns:", list(table.columns))
        print(table.head(1).to_string(index=False))
        for col in ["edit", "intervention", "scenario", "dose_edit"]:
            if col in table:
                numeric = table.select_dtypes(include="number").columns
                print(table.groupby(col)[list(numeric)].mean().to_string())
                break

print("=== AUDIT STAYS ===")
print(list(audit_ids236))
print("=== FIT COMPONENT SHAPES ===")
for key in ["T0", "scale", "GF", "RF", "GD", "RD"]:
    obj = st242[key]
    print(key, compact240(obj))
```
