The visible factual checks show increases of 0.00411 and 0.01019 nats per reading; the second exceeds the limit. I won’t amplify this edit. Next I’ll recover the aggregate result and the safer candidate’s exact construction, so I can reduce factual damage without discarding the insulin improvement.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== COMPACT FACTUAL AUDIT ===")
paired298 = []
for k in range(38, 42):
    p = Path(f"/work/fwd_{k}")
    a = pd.read_csv(p / "factual_original.csv")
    b = pd.read_csv(p / "factual_edited.csv")
    assert a[["stay_id", "hour", "event"]].equals(b[["stay_id", "hour", "event"]])
    q = a[["stay_id", "event"]].copy()
    q["original"] = a.nll
    q["edited"] = b.nll
    q["delta"] = b.nll - a.nll
    paired298.append(q)
    print(k, "n", len(q), "original", round(a.nll.mean(), 8),
          "edited", round(b.nll.mean(), 8), "delta", round(q.delta.mean(), 8))
all298 = pd.concat(paired298, ignore_index=True)
print("ALL", len(all298), "stays", all298.stay_id.nunique(),
      "delta", float(all298.delta.mean()))
print("BY EVENT", all298.groupby("event").delta.mean().round(7).to_dict())
stay298 = all298.groupby("stay_id").agg(delta_sum=("delta", "sum"), n=("delta", "size"))
rng298 = np.random.default_rng(298)
boot298 = []
for _ in range(2000):
    ix = rng298.integers(0, len(stay298), len(stay298))
    x = stay298.iloc[ix]
    boot298.append(x.delta_sum.sum() / x.n.sum())
print("stay-bootstrap delta quantiles", np.quantile(boot298, [.025, .5, .975]))

current298 = {n: torch.as_tensor(load(n)).float().clone() for n in edited()}
print("\n=== EDIT OBJECTS ===")
objnames298 = sorted(k for k in list(globals())
                     if any(s in k.lower() for s in ["286", "292", "295", "296", "236"])
                     and not k.startswith("_"))
print("names:", ", ".join(objnames298)[:1600])
for k in ["B286", "fits286", "candidate295", "candidate296",
          "checkpoint292", "safe236", "P296", "gains296"]:
    if k not in globals():
        continue
    v = globals()[k]
    print(k, type(v).__name__, getattr(v, "shape", ""))
    if isinstance(v, dict):
        print("keys", list(v.keys())[:16])
        for kk, vv in list(v.items())[:4]:
            if isinstance(vv, dict):
                print(" ", kk, "dict keys", list(vv.keys())[:12])
            else:
                print(" ", kk, type(vv).__name__, getattr(vv, "shape", ""),
                      str(vv)[:100] if np.isscalar(vv) else "")
    elif np.isscalar(v):
        print(v)

print("\n=== SELECTED NOTE EVIDENCE ===")
text298 = Path("/work/notes.md").read_text()
parts298 = text298.split("\n## ")
chosen298 = []
for s in parts298:
    low = s.lower()
    if any(w in low for w in ["identification", "exogen", "randomiz",
                              "six-lag", "6-lag", "distributed lag"]):
        chosen298.append(s)
budget298 = 3700
for s in (chosen298[:5] + chosen298[-4:]):
    excerpt = s[:min(650, budget298)]
    print("\n##", excerpt)
    budget298 -= len(excerpt)
    if budget298 <= 0:
        break

print("\n=== RECENT SUMMARY METADATA ===")
for k in [34, 35, 36, 38, 39, 40, 41]:
    p = Path(f"/work/fwd_{k}/summary.json")
    if p.exists():
        s = json.loads(p.read_text())
        print(k, "keys", list(s.keys()),
              "results", str(s.get("results", {}))[:380])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 298\nCompact audit of Forward38–41 and saved all current "
            "weights as current298. Current mixed candidate factual delta over "
            f"{len(all298)} readings: {all298.delta.mean():.8f}; "
            f"stay-bootstrap 95% interval {np.quantile(boot298,[.025,.975]).tolist()}. "
            "Recovering safer projection construction before final adjustment. "
            "No weights changed. Counts: 298 Python runs, 42 forwards.\n")
```
