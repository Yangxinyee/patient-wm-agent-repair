The latest rollout strengthens the insulin response, but norepinephrine still has little effect. These are model predictions, not evidence of the true effects. I’ll recover the missing factual checks and record-based estimates before changing anything; exceeding the 0.005-NLL limit would rule out the current edit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== FORWARD 38–42 FILES AND FACTUAL RESULTS ===")
for k in range(38, 43):
    folder = Path(f"/work/fwd_{k}")
    print(k, [p.name for p in folder.iterdir()] if folder.exists() else "missing")
    for p in sorted(folder.glob("factual_*.csv")):
        df297 = pd.read_csv(p)
        print(p.name, "rows", len(df297), "columns", df297.columns.tolist())
        for c in df297.columns:
            if "nll" in c.lower() or "loss" in c.lower():
                v = pd.to_numeric(df297[c], errors="coerce")
                print(c, "count", int(v.notna().sum()), "mean", float(v.mean()))
        for c in ("stay", "stay_id"):
            if c in df297:
                print("stay_ids", sorted(df297[c].unique().tolist()))

print("\n=== CURRENT EDIT SIZES ===")
for name in edited():
    a297 = torch.as_tensor(load(name)).float()
    b297 = torch.as_tensor(load_original(name)).float()
    print(name, tuple(a297.shape), "delta_norm", float((a297-b297).norm()))

print("\n=== RELEVANT KERNEL OBJECTS ===")
keys297 = sorted(k for k in list(globals())
                 if any(s in k.lower() for s in
                        ["286", "292", "295", "296", "236", "effect",
                         "causal", "lagfit", "target", "checkpoint"]))
for k in keys297:
    obj297 = globals()[k]
    text297 = f"{k}: {type(obj297).__name__}"
    if hasattr(obj297, "shape"):
        text297 += f" shape={obj297.shape}"
        if np.prod(obj297.shape) <= 30:
            text297 += f" values={obj297}"
    elif isinstance(obj297, dict):
        text297 += f" keys={list(obj297.keys())[:25]}"
    elif isinstance(obj297, (str, float, int, tuple)):
        text297 += f" value={str(obj297)[:300]}"
    print(text297)

print("\n=== RECORD-EFFECT NOTE EXCERPTS ===")
notes297 = Path("/work/notes.md").read_text()
sections297 = notes297.split("\n## ")
matches297 = [(i, s) for i, s in enumerate(sections297)
              if any(t in s.lower() for t in
                     ["conditional-exog", "conditional exog", "identification",
                      "six-lag", "6-lag", "distributed lag", "randomiz", "instrument"])]
for i, s in matches297[-12:]:
    print("\n## " + s[:1300])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 297\nRestart recovery after Forward42: current rollout shifts "
            "[-4.754, 2.098, .329, -.395]. Recovering architecture, Forward38–41 "
            "factual gates, exact edit objects, and record-derived effect evidence. "
            "No weights changed. Counts: 297 Python runs, 42 forwards.\n")
```
