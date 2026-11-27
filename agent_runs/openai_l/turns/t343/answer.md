The last cohort’s factual increase is **0.00417**; all four cohorts pass, with a pooled increase around **0.0042**. This gives us an audited fallback. I’ll test a precomputed head with stronger norepinephrine response and nearly unchanged insulin slopes. It is worth keeping only if rollout accuracy improves and pooled factual fit remains within the limit.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import re

current_audit295 = {}
audit_groups295 = []
for folder295 in ["fwd_35", "fwd_31", "fwd_36", "fwd_33"]:
    base295 = Path("/work") / folder295
    o295 = pd.read_csv(base295 / "factual_original.csv")
    e295 = pd.read_csv(base295 / "factual_edited.csv")
    cols295 = [c for c in o295.columns if "nll" in c.lower()]
    assert cols295, (folder295, list(o295.columns))
    col295 = "nll" if "nll" in cols295 else cols295[0]
    assert len(o295) == len(e295)
    delta295 = float(e295[col295].mean() - o295[col295].mean())
    current_audit295[folder295] = (len(o295), delta295)
    print(folder295, "n=", len(o295), "delta=", delta295)
    if "stay_id" in o295.columns:
        og295 = o295.groupby("stay_id")[col295].agg(["sum", "count"])
        eg295 = e295.groupby("stay_id")[col295].agg(["sum", "count"])
        assert og295["count"].equals(eg295["count"])
        audit_groups295.append(pd.DataFrame({
            "dsum": eg295["sum"] - og295["sum"],
            "n": og295["count"]
        }))

n295 = sum(v[0] for v in current_audit295.values())
pooled295 = sum(v[0]*v[1] for v in current_audit295.values()) / n295
print("CURRENT POOLED DELTA:", pooled295, "readings:", n295)
if len(audit_groups295) == 4:
    g295 = pd.concat(audit_groups295)
    residual295 = g295["dsum"] - pooled295*g295["n"]
    se295 = np.sqrt(len(g295)/(len(g295)-1) * np.sum(residual295**2)) / g295["n"].sum()
    print("Stay-cluster SE:", se295, "one-sided 95% upper:", pooled295 + 1.645*se295)

text295 = Path("/work/notes.md").read_text()
print("=== LATER TARGET-AUDIT NOTES ===")
for s295 in text295.split("\n## "):
    title295 = s295.splitlines()[0] if s295 else ""
    if re.search(r"\b(83|97)\b", title295):
        print(s295[:1400])

print("=== LATER TARGET OBJECT STRUCTURE ===")
for name295 in ["cf_targets83", "alpha_targets97"]:
    obj295 = globals()[name295]
    for key295 in [("insulin", 2.0), ("norepinephrine", 2.0)]:
        v295 = obj295[key295]
        if isinstance(v295, dict):
            print(name295, key295, {
                k: (float(x) if np.isscalar(x) else
                    {"type": type(x).__name__, "shape": getattr(x, "shape", None)})
                for k, x in v295.items()
            })
        else:
            print(name295, key295, type(v295).__name__, getattr(v295, "shape", None))

backup295 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
candidate295_key = (.35, .5, 4.0)
candidate295 = grid286[candidate295_key]
T295 = candidate295["T"]
assert T295.shape == (193, 17) and np.isfinite(T295).all()
save_edit("attr_heads.value_bin.2.weight", torch.tensor(T295[:192].T.copy(), dtype=torch.float32))
save_edit("attr_heads.value_bin.2.bias", torch.tensor(T295[192].copy(), dtype=torch.float32))
print("STAGED:", candidate295_key, "offline score:", candidate295["score"])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 295 / complete fallback audit and NE alternative\n"
        f"backup290/grid286(.35,.3,8) audited on all400 tuning stays: "
        f"{n295} readings, pooled factual delta {pooled295:.8f}. "
        f"Cohort deltas {current_audit295}. Preserved in backup295. "
        "Staged grid286(.35,.5,4), changing only final value-head weights/bias. "
        f"Offline score {candidate295['score']}. GPU validation pending.\n"
    )
```