```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== CURRENT OPERATIONS ===")
print(ops())

print("\n=== FORWARD 26 FILES ===")
for p in sorted(Path("/work/fwd_26").iterdir()):
    print(p.name)
    if p.suffix == ".json":
        print(p.read_text()[:5000])
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        print("columns:", list(df.columns))
        print(df.head(2).to_string(index=False))
        for c in df.columns:
            if "nll" in c.lower():
                print(c, "mean =", df[c].mean())

print("\n=== EARLY NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:15500])

print("\n=== AVAILABLE KERNEL VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"notes", "df", "p", "v"}:
        continue
    if callable(v):
        if getattr(v, "__module__", "") == "__main__":
            print(k, "function")
    elif isinstance(v, np.ndarray):
        print(k, "array", v.shape)
    elif isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape)
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", list(v)[:8])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Python run 67\n"
            "- Prior-head ablation rollout remains weak: insulin -2.273/+1.446, NE +0.064/-0.128.\n"
            "- Recovering architecture, factual pass 26, and early diagnostics before deciding the next edit.\n"
            "- Forward passes used: 27; current edits unchanged pending inspection.\n")
```
