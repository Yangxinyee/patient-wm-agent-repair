The exact rollout returned zero for all four edited-model shifts. That confirms dose invariance, not the true effects.

I’ll record the result, calculate uncertainty around the paired factual NLL change, and inspect the history features for later-hour causal checks. A reproducible nonzero effect in those checks would challenge the near-zero target despite the successful edit.

```python
import numpy as np
import pandas as pd
from scipy.stats import t
import json

factual_original_2 = pd.read_csv("/work/fwd_2/factual_original.csv")
factual_edited_2 = pd.read_csv("/work/fwd_2/factual_edited.csv")
print("Factual columns:", factual_original_2.columns.tolist())
print(factual_original_2.head(2).to_string(index=False))

assert len(factual_original_2) == len(factual_edited_2)
for key in ("stay_id", "hour", "event", "code", "value"):
    if key in factual_original_2.columns:
        assert factual_original_2[key].equals(factual_edited_2[key]), key

nll_candidates = [
    c for c in factual_original_2.columns if "nll" in c.lower()
]
nll_col = next(
    (c for c in ("nll", "value_nll", "nll_value", "fine_bin_nll")
     if c in nll_candidates),
    nll_candidates[0]
)
print("Using NLL column:", nll_col)

paired_factual_2 = pd.DataFrame({
    "stay_id": factual_original_2["stay_id"],
    "original_nll": factual_original_2[nll_col],
    "edited_nll": factual_edited_2[nll_col],
})
paired_factual_2["delta"] = (
    paired_factual_2["edited_nll"] - paired_factual_2["original_nll"]
)
paired_factual_2 = paired_factual_2.loc[
    np.isfinite(paired_factual_2["delta"])
].copy()

def paired_nll_summary(table):
    delta_mean = float(table["delta"].mean())
    clusters = table.groupby("stay_id")["delta"].agg(["sum", "count"])
    scores = clusters["sum"] - clusters["count"] * delta_mean
    n_clusters = len(clusters)
    se = float(np.sqrt(
        n_clusters / (n_clusters - 1) * np.square(scores).sum()
    ) / len(table))
    margin = float(t.ppf(0.975, n_clusters - 1) * se)
    return {
        "n_readings": len(table), "n_stays": n_clusters,
        "original_nll": float(table["original_nll"].mean()),
        "edited_nll": float(table["edited_nll"].mean()),
        "delta": delta_mean, "cluster_se": se,
        "ci95": [delta_mean - margin, delta_mean + margin],
    }

factual_summary_2 = paired_nll_summary(paired_factual_2)
print("Paired factual summary:", factual_summary_2)

event_col = next(
    (c for c in ("event", "code") if c in factual_original_2.columns), None
)
if event_col:
    paired_factual_2[event_col] = factual_original_2.loc[
        paired_factual_2.index, event_col
    ]
    print("By event:")
    for event, rows in paired_factual_2.groupby(event_col):
        print(event, paired_nll_summary(rows))

rollout_edited_3 = pd.read_csv("/work/fwd_3/dose_edit_edited.csv")
print("\nEdited rollout, reached stay-hours:")
for edit, rows in rollout_edited_3.loc[
    rollout_edited_3["reached"].astype(bool)
].groupby("edit"):
    print(edit, "n:", len(rows),
          "mean:", float(rows["shift"].mean()),
          "max absolute:", float(rows["shift"].abs().max()))

print("\n=== EXISTING HISTORY FEATURES ===")
print(feature_names)
print("Compact-array channels:", events)
print("First stay hour grid:", hour[0].tolist())
print("First five compact rows:")
print(arr[0, :5, :])
print("First five compact dose rows:")
print(dose_arr[0, :5, :])
print("First treatment hours:", hours_all[:24].tolist())
print("First NE treatment vector:", Aall[:24].tolist())
print("Treatment residual SD, positive late-hour doses:",
      float(np.std(ra[(Aall > 0) & (hours_all >= 12)])))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward 2 and 3 validation, first 100 tuning stays\n"
        "Factual original NLL 1.50155; candidate 1.50136 "
        "(approximately -0.00019 nats). Paired cluster summary:\n"
        + json.dumps(factual_summary_2) + "\n"
        "Exact evaluation rollout: insulin x2 and half, NE x2 and half "
        "all have zero shift on every reached stay-hour. Original means "
        "were +0.113, -0.186, +0.050, +0.013 respectively. This confirms "
        "behavioral invariance, not causal truth. Need validate factual fit "
        "on remaining tuning stays and check causal estimates specifically "
        "at later ICU hours, using only pre-dose history.\n"
    )
```