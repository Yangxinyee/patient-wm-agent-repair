The reserved stays reproduce a small NLL increase, +0.00346. That is encouraging, but the narrow safety margin is not worth a likely small extra dose response. I’ll pool the losses and lower recurrent gain to 1.30. Lower factual loss with nearly unchanged drug responses would confirm this tradeoff.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

def factual_pairs(k):
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    keys = ["stay_id", "hour", "event", "observed"]
    assert o[keys].equals(e[keys])
    p = o[["stay_id", "event"]].copy()
    p["delta_nll"] = e["nll"].to_numpy() - o["nll"].to_numpy()
    return p

def loss_audit(p):
    delta = float(p["delta_nll"].mean())
    c = p.groupby("stay_id")["delta_nll"].agg(["sum", "count"])
    g, n = len(c), len(p)
    se = float(np.sqrt(g/(g-1) *
                      np.square(c["sum"] - delta*c["count"]).sum()/n**2))
    return {"stays": g, "readings": n, "delta_nll": delta,
            "cluster_se": se, "one_sided_95pct_upper": delta + 1.645*se}

pooled135 = pd.concat([factual_pairs(43), factual_pairs(45)],
                      ignore_index=True)
print("=== MEASURED GAIN-1.35 LOSS ON 200 STAYS ===")
print(loss_audit(pooled135))
print(pooled135.groupby("event")["delta_nll"].agg(["mean", "count"]))

# Local interpolation is a planning diagnostic, not a measured validation.
d0 = factual_pairs(9)["delta_nll"].to_numpy()    # numeric gain 4 alone
d035 = factual_pairs(43)["delta_nll"].to_numpy()
d1 = factual_pairs(38)["delta_nll"].to_numpy()  # recurrent gain 2
assert len(d0) == len(d035) == len(d1)
a113 = ((d035-d0) - 0.35**2*(d1-d0)) / (0.35-0.35**2)
b113 = (d1-d0) - a113
for gain in [1.25, 1.30, 1.35]:
    t = gain-1
    print("Interpolated probe delta NLL, gain", gain,
          float((d0 + t*a113 + t*t*b113).mean()))

before113 = ops()
assert len(before113) == 2
assert before113[-1]["name"] == "core.b_in.weight"
assert abs(before113[-1]["c"] - 1.35) < 1e-10
Path("/work/recurrent_135_confirmed_113.json").write_text(
    json.dumps(before113, indent=2)
)
undo()
scale("core.b_in.weight", 1.30)
Path("/work/recurrent_130_113.json").write_text(
    json.dumps(ops(), indent=2)
)
print("\n=== INSTALLED FINAL-CANDIDATE OPERATIONS ===")
print(ops())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 113 / forward 45: conservative recurrent gain\n"
        f"- Gain 1.35 pooled measured loss audit: {loss_audit(pooled135)}.\n"
        "- Validation factual increase approximately +0.00346, agreeing with probe.\n"
        "- Reduced recurrent gain to 1.30 to preserve more factual safety margin.\n"
        "- Numeric gain 4 unchanged. Local NLL interpolation is only a planning diagnostic.\n"
        "- Gain-1.30 factual fit and rollouts require direct validation.\n"
        "- Forward passes used: 45; 5 remain.\n"
    )
```