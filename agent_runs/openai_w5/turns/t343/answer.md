The failed request contained duplicate stay IDs, so it provided no new evidence. I’ll recover the unique fresh-test stays and current checkpoint, then repeat the paired factual test. A loss increase above 0.005 nats would rule out this candidate.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRECENT FORWARD FILES")
for k in range(34, 38):
    p = Path(f"/work/fwd_{k}")
    if not p.exists():
        continue
    print(p.name, [x.name for x in p.iterdir()])
    for model in ("original", "edited"):
        f = p / f"factual_{model}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        print(model, "shape:", d.shape, "columns:", d.columns.tolist())
        idcols = [c for c in d.columns if "stay" in c.lower()]
        if idcols:
            ids = d[idcols[0]].drop_duplicates().tolist()
            print("unique stays:", ids)
        print(d.head(2).to_string(index=False))

print("\nRELEVANT KERNEL VARIABLES")
keys = sorted(k for k in list(globals())
              if not k.startswith("_")
              and any(s in k.lower() for s in
                      ("base289", "stay", "target", "beta", "tau", "strength", "center")))
for k in keys[:100]:
    v = globals()[k]
    if isinstance(v, (str, int, float, np.number)):
        desc = repr(v)[:180]
    elif isinstance(v, dict):
        desc = f"dict, {len(v)} keys; first keys={list(v)[:12]}"
    elif hasattr(v, "shape"):
        desc = f"{type(v).__name__}, shape={v.shape}"
    elif isinstance(v, (list, tuple)):
        desc = f"{type(v).__name__}, len={len(v)}; first={repr(v[:8])[:180]}"
    else:
        desc = type(v).__name__
    print(k, desc)

print("\nEARLY NOTEBOOK")
notes291 = Path("/work/notes.md").read_text()
print(notes291[:7000])
print("\nCURRENT EDIT COUNT:", len(edited()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 291: recover after duplicate-ID request failure\n"
            "- Failed forward request provided no evidence and was not counted.\n"
            "- Recovering unique fresh-test stays and the current tau=1250 baseline.\n"
            "- No weights changed; fresh-stay paired factual validation remains required.\n")
```