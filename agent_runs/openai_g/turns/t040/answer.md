The final batch increased NLL by 0.00022 nats, still comfortably within the limit. I’ll compute the full 400-stay difference and its uncertainty, then inspect the existing causal-analysis data. For the norepinephrine question, a consistent response across the evaluated hours would matter more than one isolated lag.

```python
import pandas as pd
import numpy as np
from pathlib import Path

paired_parts = []
for folder in sorted(Path("/work").glob("fwd_*")):
    op, ep = folder / "factual_original.csv", folder / "factual_edited.csv"
    if not (op.exists() and ep.exists()):
        continue
    fo, fe = pd.read_csv(op), pd.read_csv(ep)
    assert fo[["stay_id", "hour", "event", "observed"]].equals(
        fe[["stay_id", "hour", "event", "observed"]]
    )
    pp = fo[["stay_id", "hour", "event"]].copy()
    pp["original_nll"] = fo["nll"].to_numpy()
    pp["edited_nll"] = fe["nll"].to_numpy()
    pp["delta"] = pp.edited_nll - pp.original_nll
    paired_parts.append(pp)

factual_all_400 = pd.concat(paired_parts, ignore_index=True)
assert set(factual_all_400.stay_id.unique()) == set(tuning_ids_all)
assert not factual_all_400.duplicated(["stay_id", "hour", "event"]).any()

dmean = factual_all_400.delta.mean()
cluster_score = (
    factual_all_400.assign(score=factual_all_400.delta - dmean)
    .groupby("stay_id").score.sum().to_numpy()
)
G = len(cluster_score)
dse = np.sqrt(G / (G - 1) * np.sum(cluster_score ** 2)) / len(factual_all_400)
factual_full_summary = {
    "stays": G, "readings": len(factual_all_400),
    "original_nll": float(factual_all_400.original_nll.mean()),
    "edited_nll": float(factual_all_400.edited_nll.mean()),
    "delta": float(dmean), "stay_cluster_se": float(dse),
    "ci95": [float(dmean - 1.96*dse), float(dmean + 1.96*dse)]
}
print("=== FULL FACTUAL VALIDATION ===")
print(factual_full_summary)
print(factual_all_400.groupby("event")[["original_nll", "edited_nll", "delta"]].mean())

print("=== CAUSAL ANALYSIS OBJECTS ===")
for key in [
    "ds", "dis", "g", "gg", "gi", "events", "drug_records", "feature_names",
    "Xall", "Aall", "Yall", "hours_all", "groups_all",
    "X_ne_v2", "A_ne_v2", "ne_features_v2", "positive_ne_v2",
    "ne_late_residuals_v3", "ne_residuals_v2", "map_baseline_v3",
    "treatment_hours_v2", "eligible"
]:
    ob = globals().get(key)
    desc = {"name": key, "type": type(ob).__name__}
    if hasattr(ob, "shape"):
        desc["shape"] = ob.shape
    if isinstance(ob, dict):
        desc["keys"] = list(ob)[:15]
    if isinstance(ob, (list, tuple)):
        desc["length"] = len(ob)
        desc["sample"] = str(ob[:3])[:250]
    if isinstance(ob, pd.DataFrame):
        desc["columns"] = list(ob.columns)[:20]
    print(desc)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Full 400-stay factual validation\n")
    f.write(str(factual_full_summary) + "\n")
    f.write("All four paired batches use unchanged three-tensor edit. "
            "The CI measures tuning sampling uncertainty, not a guarantee "
            "for unseen evaluation stays. No weight changes in this run.\n")
```