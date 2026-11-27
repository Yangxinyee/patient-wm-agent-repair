The second fresh check increased factual NLL by only about +0.00221. Both additional checks support retaining the stronger edit, whose main insulin response matches the record-based target closely. I’ll verify that the submitted weights are exactly those tested and record the final metrics; a weight mismatch or factual-budget violation would overturn this choice.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
import json

def final_factual_audit245(forwards):
    parts = []
    for k in forwards:
        o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
        e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
        keys = ["stay_id", "hour", "event", "observed"]
        assert len(o) == len(e)
        assert o[keys].equals(e[keys])
        assert np.isfinite(o["nll"]).all() and np.isfinite(e["nll"]).all()
        parts.append(pd.DataFrame({
            "stay_id": o["stay_id"],
            "original": o["nll"],
            "edited": e["nll"],
            "delta": e["nll"] - o["nll"]
        }))
    d = pd.concat(parts, ignore_index=True)
    mu = float(d["delta"].mean())
    g = d.groupby("stay_id")["delta"].agg(["sum", "count"])
    score = g["sum"] - mu * g["count"]
    se = float(np.sqrt(
        len(g) / (len(g) - 1) * np.square(score).sum() / len(d)**2
    ))
    return {
        "stays": int(len(g)),
        "readings": int(len(d)),
        "original_nll": float(d["original"].mean()),
        "edited_nll": float(d["edited"].mean()),
        "delta": mu,
        "stay_cluster_se": se,
        "descriptive_one_sided_95_upper": mu + 1.645 * se
    }

final_factual245 = {
    "held_out_tuning": final_factual_audit245([44, 45, 46, 47]),
    "fresh_training_checks": final_factual_audit245([49, 50]),
    "pooled_descriptive": final_factual_audit245([44, 45, 46, 47, 49, 50])
}
assert final_factual245["held_out_tuning"]["delta"] < 0.005
assert final_factual245["fresh_training_checks"]["delta"] < 0.005

assert set(edited()) == {"core.b_in.weight"}
accepted_B245 = np.asarray(load("core.b_in.weight"), dtype=np.float32).copy()
assert np.array_equal(accepted_B245, np.asarray(candidate_B237))
assert np.isfinite(accepted_B245).all()
final_sha245 = hashlib.sha256(accepted_B245.tobytes()).hexdigest()

dose_final245 = pd.read_csv("/work/fwd_48/dose_edit_edited.csv")
final_readouts245 = (
    dose_final245.loc[dose_final245["reached"].astype(bool)]
    .groupby("edit")["shift"].mean().to_dict()
)

print("FINAL CHOICE: retain tested candidate_B237; no new weight changes.")
print("Edited tensors:", edited())
print("SHA256:", final_sha245)
print("FACTUAL AUDIT:")
print(json.dumps(final_factual245, indent=2))
print("MATCHED TUNING ROLLOUT READOUTS:")
print(json.dumps(final_readouts245, indent=2))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run245: final audit and submission decision\n"
        "Retain candidate_B237 exactly as tested in forward48 and factual "
        "forwards44–47,49–50. Only core.b_in.weight is edited: original + "
        "1.40*delta_ins218 + 1.00*delta_ne218. No half-dose decoder blend, "
        "no gradient training, and no further weight changes in this run.\n"
        "SHA256 of submitted float32 tensor: " + final_sha245 + "\n"
        "Factual audit: " + json.dumps(final_factual245) + "\n"
        "Matched 99-stay tuning rollout readouts: "
        + json.dumps(final_readouts245) + "\n"
        "Main insulin response closely matches the training-record-derived "
        "target on this cohort. Insulin-half and NE magnitudes remain smaller "
        "than their estimated targets. Drug slopes and accumulated effects "
        "depend on identification and persistence assumptions; none are "
        "revealed simulator truth. Additional training checks support factual "
        "stability but are not independent held-out tuning evidence. "
        "Held-out tuning empirical factual increase is below +0.005; "
        "unseen-stay performance is not guaranteed.\n"
        "All 50 forward passes used; 245 Python runs used. Final decision "
        "is the stronger tested edit, not fallback_B237.\n"
    )
```