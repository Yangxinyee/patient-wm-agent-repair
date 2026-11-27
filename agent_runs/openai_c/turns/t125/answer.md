The selective latent-head gain barely changed the dose responses; they remain far below the effects estimated from the records. I’ll inspect its factual fit and recover the architecture and saved diagnostics. A useful next edit must strengthen dose sensitivity, not merely alter factual predictions.

```python
from pathlib import Path
import json, numpy as np, pandas as pd

print("CURRENT OPERATIONS")
print(ops())

print("\nARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nFORWARD 33 FILES")
for p in sorted(Path("/work/fwd_33").iterdir()):
    print(p.name)
    if p.suffix == ".json":
        print(p.read_text()[:12000])
    elif p.name.startswith("factual_") and p.suffix == ".csv":
        d = pd.read_csv(p)
        print(d.columns.tolist(), d.shape)
        print(d.head(2).to_string(index=False))
        for c in d.columns:
            if "nll" in c.lower():
                print(c, pd.to_numeric(d[c], errors="coerce").mean())

print("\nKERNEL INVENTORY")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if callable(v):
        if getattr(v, "__module__", "") == "__main__":
            print(k, "function")
    elif isinstance(v, np.ndarray):
        print(k, "array", v.shape)
    elif isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape)
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:12])
    elif isinstance(v, (list, tuple)) and len(v) < 120:
        print(k, type(v).__name__, str(v)[:400])
    elif isinstance(v, (str, int, float)) and len(str(v)) < 300:
        print(k, repr(v))

print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:15000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 88: restart recovery\n"
            "- Forward 34 selective prior-head gain: insulin -2.428/+1.538, NE +0.166/+0.024; no material improvement.\n"
            "- Recovering factual pass 33, architecture, and saved estimators before choosing a new pathway.\n"
            "- Current edits unchanged pending inspection. Forward passes used: 34; 16 remain.\n")
```