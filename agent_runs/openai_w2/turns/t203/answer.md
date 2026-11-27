The rollout strengthened norepinephrine’s response to **+5.43 / −2.33**, while insulin stayed essentially unchanged at **−14.13 / +6.82**. That supports the edit’s selectivity. I’ll check stay-level aggregation and recover the causal evidence before deciding whether more strengthening is justified.

```python
import pandas as pd
import numpy as np
from pathlib import Path

validated_B176 = np.asarray(load("core.b_in.weight"), dtype=np.float32).copy()
validated_W176 = trial_W175.copy()

d176 = pd.read_csv("/work/fwd_21/dose_edit_edited.csv")
print("Rollout columns:", list(d176.columns))
print(d176.head(4).to_string(index=False))
if {"reached", "edit", "stay_id", "shift"}.issubset(d176.columns):
    reached176 = d176["reached"].map(
        lambda x: str(x).lower() in {"true", "1", "1.0"})
    stay_means176 = (
        d176.loc[reached176]
        .groupby(["edit", "stay_id"])["shift"].mean()
    )
    print("\nPer-stay aggregated rollout shifts:")
    print(stay_means176.groupby("edit").agg(["mean", "std", "count"]))

print("\nPrevious target-alignment columns:", list(aligned166.columns))
print(aligned166.head(3).to_string(index=False))

independent176 = pd.read_csv("/work/fwd_18/factual_original.csv")
print("\nIndependent factual cohort IDs:")
print(sorted(independent176["stay_id"].unique().tolist()))

notes176 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-BASED EVIDENCE")
print(notes176[:7000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward20–21 / run176: reduced NE enhancement passes locally\n"
            "Only core.b_in.weight edited, hash 4a7d16d8f6. Factual NLL "
            "increase +0.00373 on 99 tuning stays. Pooled reached-hour "
            "rollout shifts: insulin x2/half -14.134/+6.816; NE "
            "x2/half +5.428/-2.333. Saved exact weights as validated_B176 "
            "and decoder as validated_W176. Insulin response is stable; "
            "NE response increases. Require independent factual validation "
            "and record-based target audit before final selection.\n")
```