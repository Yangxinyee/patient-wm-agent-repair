The second 100 stays show an NLL increase of about 0.00018—well below the 0.005 limit and roughly balancing the first batch’s improvement. I’ll combine the paired results and inspect the history features for late-hour causal checks. Those checks must remain near zero to justify removing dose sensitivity.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

def paired_summary_v2(table):
    n = len(table)
    delta = float(table["delta"].mean())
    by_stay = table.groupby("stay_id")["delta"].agg(["sum", "count"])
    G = len(by_stay)
    scores = by_stay["sum"].to_numpy() - delta * by_stay["count"].to_numpy()
    se = float(np.sqrt(G / (G - 1) * np.sum(scores**2)) / n) if G > 1 else np.nan
    return {"n_readings": n, "n_stays": G,
            "original_nll": float(table["nll_original"].mean()),
            "edited_nll": float(table["nll_edited"].mean()),
            "delta": delta, "cluster_se": se,
            "ci95": [delta - 1.96 * se, delta + 1.96 * se]}

factual_pairs = []
for k in (2, 4):
    original = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    candidate = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    keys = ["stay_id", "hour", "event"]
    paired = original[keys + ["nll"]].merge(
        candidate[keys + ["nll"]], on=keys, suffixes=("_original", "_edited"),
        validate="one_to_one")
    assert len(paired) == len(original) == len(candidate)
    paired["delta"] = paired["nll_edited"] - paired["nll_original"]
    factual_pairs.append(paired)
    print("Batch", k, paired_summary_v2(paired))

factual_pairs_200 = pd.concat(factual_pairs, ignore_index=True)
summary_200 = paired_summary_v2(factual_pairs_200)
print("Combined:", summary_200)

print("\n=== EXISTING FEATURE NAMES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (list, tuple)) and 31 <= len(obj) <= 100:
        if all(isinstance(v, str) for v in obj):
            print(name, obj)
print("Regression settings:", kwargs)
print("Dose residual consistency:",
      float(np.nanmax(np.abs(ra - (Aall - pa)))))
print("First cached treatment values:", Aall[:8].tolist())
print("First compact NE rates:", arr[0, 1:9, 5].tolist())
print("First compact NE doses:", dose_arr[0, 1:9, 1].tolist())

print("\n=== LATE-HOUR EXPOSURES ===")
for j, drug in [(4, "insulin"), (5, "norepinephrine")]:
    rates = arr[:, 13:25, j].reshape(-1)
    positive = rates[np.isfinite(rates) & (rates > 0)]
    print(drug, "positive late-hour rows:", len(positive),
          "rate quantiles:", np.quantile(positive, [0, .1, .5, .9, 1]).tolist())
print("\n=== DOSE-NULL CONSTRUCTION STATISTICS ===")
for key, val in dose_null_stats.items():
    print(key, "n:", val["n"], "mean_z:", val["mean_z"])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Factual validation, first 200 tuning stays\n"
            + json.dumps(summary_200) + "\n"
            "Second batch remains far inside the +0.005 nats tolerance. "
            "Next: late-hour causal checks, retaining only pre-infusion covariates.\n")
```
