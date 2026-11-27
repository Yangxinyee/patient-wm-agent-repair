The latest check shows a factual NLL increase of **0.00271 nats**, below the 0.005 limit on these 100 stays. I’ll verify the installed edits and combine comparable checks before spending the last two passes. Consistent results across independent stays would support keeping this candidate.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nCURRENT OPERATIONS")
print(ops())
print("\nSAVED OPERATIONS")
print(Path("/work/ops.json").read_text())

print("\nNOTEBOOK TAIL")
print(Path("/work/notes.md").read_text()[-18000:])

print("\nRECENT FORWARD FILES")
for k in range(43, 49):
    folder = Path(f"/work/fwd_{k}")
    print(k, sorted(p.name for p in folder.iterdir()))
    for filename in ("factual_original.csv", "factual_edited.csv",
                     "dose_edit_original.csv", "dose_edit_edited.csv"):
        file = folder / filename
        if file.exists():
            frame = pd.read_csv(file)
            print(filename, "shape:", frame.shape,
                  "columns:", frame.columns.tolist())
            print(frame.head(2).to_string(index=False))

print("\nRELEVANT RETAINED VARIABLES")
for name, value in list(globals().items()):
    if any(s in name.lower() for s in ("probe", "valid", "stay", "target", "gain", "audit")):
        if isinstance(value, np.ndarray):
            print(name, "array", value.shape, str(value.ravel()[:12]))
        elif isinstance(value, (list, tuple)):
            print(name, "length", len(value), str(value[:12]))
        elif isinstance(value, (dict, int, float, str)):
            print(name, str(value)[:1500])
```
