The saved edit uses two paired-neuron channels, one per drug. The rollout file also lets us check both row-weighted and equal-stay averages. I will measure each channel’s factual and imposed-dose activation ranges. This will test whether a selective gate adjustment could strengthen doubling responses without increasing factual error or the already-large halving responses.

```python
import numpy as np
import pandas as pd
import torch

print("=== CURRENT ROLLOUT: BOTH AGGREGATIONS ===")
r243_reached = r242_roll[r242_roll["reached"].astype(bool)]
for edit, part in r243_reached.groupby("edit"):
    print(edit, "rows", len(part), "stays", part.stay_id.nunique(),
          "row_mean", round(part["shift"].mean(), 5),
          "equal_stay_mean", round(part.groupby("stay_id")["shift"].mean().mean(), 5))

print("\n=== CONSTRUCTION VARIABLE NAMES / SHAPES ===")
r243_g = dict(globals())
for suffix in ("193", "194", "206", "232"):
    entries = []
    for n, v in sorted(r243_g.items()):
        if n.endswith(suffix) and not callable(v):
            if isinstance(v, dict):
                desc = "keys=" + repr(list(v)[:10])
            elif np.isscalar(v):
                desc = repr(v)[:100]
            else:
                desc = str(getattr(v, "shape", type(v).__name__))
            entries.append(n + ":" + desc)
    print(suffix, "; ".join(entries)[:1800])

print("\n=== FACTUAL PAIR PREACTIVATIONS ===")
r243_W = np.asarray(torch.as_tensor(load("z_post.0.weight")).cpu())
r243_b = np.asarray(torch.as_tensor(load("z_post.0.bias")).cpu())
r243_raw = np.asarray(raw_base192, dtype=np.float32)
r243_pre = r243_raw @ r243_W[[96, 184, 89, 156]].T + r243_b[[96, 184, 89, 156]]
r243_active = np.asarray(active192, dtype=bool)
for label, mask in (("NE active", r243_active), ("NE inactive", ~r243_active)):
    print(label, "n", mask.sum())
    for j, row in enumerate((96, 184, 89, 156)):
        print(" row", row, np.round(np.quantile(r243_pre[mask, j],
              [0, .05, .25, .5, .75, .95, 1]), 3))

print("\n=== COUNTERFACTUAL FIXTURE STRUCTURE ===")
print("raw_cf192", type(raw_cf192).__name__)
if isinstance(raw_cf192, dict):
    for k, v in raw_cf192.items():
        print(k, type(v).__name__, getattr(v, "shape", None),
              list(v)[:5] if isinstance(v, dict) else "")
print("matched_targets238:", repr(matched_targets238)[:1300])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 243\n"
            "Computing both rollout aggregations and factual paired-neuron gate margins. "
            "Considering selective imposed-dose amplification rather than whole-matrix "
            "scaling, which previously damaged factual fit. No weights changed.\n")
```