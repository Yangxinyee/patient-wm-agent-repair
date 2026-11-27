The architecture confirms that the value head depends on sampled prior latents and a last-reading register, not just the exported state. A mismatch there could explain the failed edit. I’ll inventory the cached calculations and record-derived effect estimates before checking reconstruction against the GPU.

```python
import numpy as np
import pandas as pd
import inspect
from pathlib import Path

print("FUNCTIONS")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            print(name, str(inspect.signature(obj))[:180])
        except Exception:
            print(name)

print("\nARRAYS AND TABLES")
for name, obj in sorted(list(globals().items())):
    if not name.startswith("_") and hasattr(obj, "shape") and not inspect.isclass(obj):
        print(name, type(obj).__name__, tuple(obj.shape))

print("\nFORWARD FILES")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\nCAUSAL RESULT TABLES")
for filename in ("dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv"):
    frame = pd.read_csv("/work/" + filename)
    print(filename, frame.shape, frame.columns.tolist())
    print(frame.head(8).to_string(index=False))

print("\nNOTE SECTION HEADINGS")
for line in Path("/work/notes.md").read_text().splitlines():
    if line.startswith("##"):
        print(line)
print("ACTIVE EDITS:", edited())
```