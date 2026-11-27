The numeric edit amplifies only the combined dose/rate coordinate; it leaves measurement inputs and presence flags untouched. I’ll now calculate the factual safety margin, recover the gain-1.30 rollout results, and identify independent stays for the final checks. A pooled upper bound below 0.005 would support retaining this edit.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

def factual_audit(pass_numbers):
    pieces = []
    for k in pass_numbers:
        folder = Path(f"/work/fwd_{k}")
        original = pd.read_csv(folder / "factual_original.csv")
        edited = pd.read_csv(folder / "factual_edited.csv")
        keys = ["stay_id", "hour", "event", "observed"]
        assert original[keys].equals(edited[keys]), f"Alignment failure: {k}"
        part = original[["stay_id", "event", "nll"]].copy()
        part["delta"] = edited["nll"] - original["nll"]
        pieces.append(part)
    rows = pd.concat(pieces, ignore_index=True)
    clusters = rows.groupby("stay_id").agg(
        delta_sum=("delta", "sum"), n=("delta", "size"))
    assert sum(p.stay_id.nunique() for p in pieces) == len(clusters)
    mean = rows.delta.mean()
    residual = clusters.delta_sum - mean * clusters.n
    se = np.sqrt(len(clusters) / (len(clusters) - 1) *
                 np.square(residual).sum()) / clusters.n.sum()
    result = {
        "passes": pass_numbers, "stays": len(clusters), "readings": len(rows),
        "delta_nll": float(mean), "cluster_se": float(se),
        "one_sided_95pct_upper": float(mean + 1.644854 * se)
    }
    return result, rows

audit_130, factual_130_rows = factual_audit([46, 48])
print("GAIN 1.30 FACTUAL AUDIT:", audit_130)
print("BY EVENT:", factual_130_rows.groupby("event").delta.mean().to_dict())

rollout47 = {}
for model in ("original", "edited"):
    df = pd.read_csv(f"/work/fwd_47/dose_edit_{model}.csv")
    reached = df.loc[df.reached.astype(bool)]
    by_stay = reached.groupby(["edit", "stay_id"]).shift.mean()
    rollout47[model] = by_stay.groupby("edit").mean().to_dict()
print("GAIN 1.30 ROLLOUT:", rollout47)

for name in ("target_train", "target_probe"):
    if name in globals():
        print(name, repr(globals()[name])[:1200])

tuning_stays = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"]).stay_id.unique().tolist())
previous_check_stays = set()
for folder in Path("/work").glob("fwd_*"):
    for filename in ("factual_original.csv", "dose_edit_original.csv"):
        path = folder / filename
        if path.exists():
            previous_check_stays.update(
                pd.read_csv(path, usecols=["stay_id"]).stay_id.unique().tolist())
final_validation_ids = [int(s) for s in tuning_stays
                        if s not in previous_check_stays][:100]
print("UNUSED CHECK STAYS:", len(final_validation_ids))
print("FINAL VALIDATION IDS:", final_validation_ids)

entry = (
    "\n## Python run 116: gain-1.30 audit\n"
    f"- Factual audit: {audit_130}\n"
    f"- Forward 47 reached-hour, within-stay-then-across-stay readouts: {rollout47}\n"
    f"- Available previously unchecked tuning stays: {len(final_validation_ids)}.\n"
    "- No edits changed. Forward passes used: 48; two remain.\n"
)
with open("/work/notes.md", "a") as f:
    f.write(entry)
```
