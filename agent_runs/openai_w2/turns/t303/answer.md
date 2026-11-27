The stronger edit predicts −14.90 for insulin doubling, almost exactly the matched record-based estimate. Halving improves to +5.94; the norepinephrine responses remain too small. Its factual increase is +0.00375, below the limit but with limited safety margin. I’ll use the remaining passes for fresh factual checks; a larger loss increase would favor reverting.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Record the exact rollout result without changing weights.
dose48_df244 = pd.read_csv("/work/fwd_48/dose_edit_edited.csv")
dose48_summary244 = (
    dose48_df244.loc[dose48_df244["reached"].astype(bool)]
    .groupby("edit")["shift"].mean().to_dict()
)
print("CANDIDATE ROLLOUT:", dose48_summary244)

# Reconstruct paired factual loss and stay-cluster uncertainty.
paired244 = []
for k in [44, 45, 46, 47]:
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    if k == 44:
        print("FACTUAL COLUMNS:", list(o.columns))
    nll_cols = [c for c in o.columns if "nll" in c.lower()]
    assert len(nll_cols) == 1, nll_cols
    nll_col = nll_cols[0]
    identity_cols = [
        c for c in o.columns
        if c in e.columns and c in
        ["stay_id", "hour", "event", "code", "value", "bundle_idx",
         "bundle_index", "minute", "timestamp"]
    ]
    assert len(o) == len(e)
    assert o[identity_cols].equals(e[identity_cols])
    paired244.append(pd.DataFrame({
        "stay_id": o["stay_id"],
        "original": o[nll_col],
        "edited": e[nll_col],
        "delta": e[nll_col] - o[nll_col]
    }))
factual244 = pd.concat(paired244, ignore_index=True)
mu244 = float(factual244["delta"].mean())
cluster244 = factual244.groupby("stay_id")["delta"].agg(["sum", "count"])
scores244 = cluster244["sum"] - mu244 * cluster244["count"]
se244 = float(np.sqrt(
    len(cluster244) / (len(cluster244) - 1)
    * np.square(scores244).sum() / len(factual244)**2
))
print("CURRENT FACTUAL:", {
    "stays": len(cluster244), "readings": len(factual244),
    "delta": mu244, "cluster_se": se244,
    "one_sided_95_upper": mu244 + 1.645 * se244
})

# Select fresh, reproducible training-stay factual checks.
train_header244 = pd.read_csv("/data/train.csv", nrows=0)
assert "stay_id" in train_header244.columns
train_ids244 = np.unique(
    pd.read_csv("/data/train.csv", usecols=["stay_id"])["stay_id"].to_numpy()
)
previous_ids244 = set()
for p in Path("/work").glob("fwd_*/*.csv"):
    if p.name.startswith(("factual_", "dose_edit_")):
        x = pd.read_csv(p, usecols=lambda c: c == "stay_id")
        if "stay_id" in x:
            previous_ids244.update(x["stay_id"].dropna().astype(int).unique())
for p in Path("/work").glob("fwd_*/states_*.npz"):
    with np.load(p, allow_pickle=False) as z:
        for key in ["stay_id", "stay_ids", "stay"]:
            if key in z.files:
                previous_ids244.update(np.unique(z[key]).astype(int).tolist())
                break
fresh_train_ids244 = np.array([
    int(s) for s in train_ids244 if int(s) not in previous_ids244
], dtype=np.int64)
assert len(fresh_train_ids244) >= 200
validation_ids244 = np.random.default_rng(20261112).choice(
    fresh_train_ids244, 200, replace=False
)
validation49_ids = sorted(validation_ids244[:100].tolist())
validation50_ids = sorted(validation_ids244[100:].tolist())
print("FORWARD49 STAYS:", validation49_ids)
print("FORWARD50 STAYS:", validation50_ids)

assert np.array_equal(
    np.asarray(load("core.b_in.weight")), np.asarray(candidate_B237)
)
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Forward48 / Run244: stronger insulin rollout validated\n"
        "Only core.b_in.weight remains edited; candidate_B237 unchanged "
        "(insulin gain 1.40, NE gain 1.00). Matched reached-hour shifts: "
        + json.dumps(dose48_summary244)
        + f". Full tuning factual delta {mu244:.9f}, clustered SE "
        f"{se244:.9f}, one-sided 95% upper {mu244 + 1.645*se244:.9f}.\n"
        "Main insulin response now agrees with the assumption-dependent "
        "record target; NE and insulin-half magnitudes remain underestimated. "
        "Selected 200 fresh training stays for two final paired factual checks. "
        "These are training-distribution checks, not new held-out tuning data. "
        "fallback_B237 remains available if safety evidence deteriorates.\n"
    )
```