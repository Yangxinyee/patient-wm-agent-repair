The direct MAP-head pilot barely changed dose sensitivity. I’ll inspect its factual result and recover the architecture and earlier identification evidence before choosing another edit; the next candidate needs a much larger rollout effect without exceeding the factual-loss limit.

```python
import os, json, numpy as np, pandas as pd
from pathlib import Path

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())
print("\nEARLY NOTEBOOK")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:19000])
print("\nPERSISTENT VARIABLE INVENTORY")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"notes_text", "key"}:
        continue
    obj = globals()[key]
    if isinstance(obj, np.ndarray):
        print(key, "array", obj.shape)
    elif isinstance(obj, pd.DataFrame):
        print(key, "DataFrame", obj.shape, list(obj.columns)[:18])
    elif isinstance(obj, dict):
        print(key, "dict keys", list(obj.keys())[:20])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function")
print("\nCURRENT OPERATIONS")
print(ops())
print("\nFORWARD 16 FILES")
for path in sorted(Path("/work/fwd_16").glob("*")):
    print(path.name)
    if path.suffix == ".json":
        print(path.read_text()[:7000])
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        print(frame.shape, frame.columns.tolist())
        print(frame.head(2).to_string(index=False))
        if "nll" in frame:
            print("mean nll", frame["nll"].mean())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Python run 31\n"
            "- Forward 17: numeric gain 4 plus direct NE head pilot gives insulin -2.521/+1.720, NE +0.129/-0.077; little benefit over numeric gain alone.\n"
            "- Recovering architecture, factual result 16, causal identification evidence, and persistent variables before selecting the next edit.\n"
            "- Python runs: 31; forward passes: 17.\n")
```
