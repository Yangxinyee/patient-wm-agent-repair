The independent batch worsens NLL by 0.00596 nats, slightly beyond the allowance, so the current edit is not yet reliably safe. I’ll remove the recurrent injection to isolate its contribution and estimate rollout effects by applying the record-derived response kernels to the recorded doses. This will distinguish a strength problem from a factual-leakage problem.

```python
from pathlib import Path
import numpy as np, pandas as pd

drop_edit("core.b_in.weight")
print("Restored safe236 baseline; current edited tensors:", edited())
print("Baseline checkpoint max discrepancy:",
      max(float(np.max(np.abs(a292(load(n)) - a292(v)))) for n, v in safe236.items()))

orig294 = pd.read_csv("/work/fwd_34/factual_original.csv")
edit294 = pd.read_csv("/work/fwd_34/factual_edited.csv")
paired294 = edit294.merge(orig294, on=["stay_id", "hour", "event"], suffixes=("_e", "_o"))
paired294["delta_nll"] = paired294["nll_e"] - paired294["nll_o"]
print("Independent current-candidate NLL delta:", paired294["delta_nll"].mean())
print(paired294.groupby("event")["delta_nll"].agg(["mean", "count"]).to_string())

kernels294 = {
    "insulin": np.array([-1.85034014, -2.25611825, -1.94263882,
                         -1.56111897, -1.02224091, -0.69018551]),
    "norepinephrine": np.array([3.35057748, 2.62101256, 1.71717035,
                               0.94251408, 0.49378644, 0.40891321])
}
roll294 = pd.read_csv("/work/fwd_33/dose_edit_edited.csv")
print("Rollout columns:", roll294.columns.tolist())
print(roll294.head(2).to_string(index=False))
drugs294 = tune293[tune293["event"].isin(kernels294)].copy()
drugs294["ihour"] = np.floor(drugs294["hour"] + 1e-7).astype(int)
drugs294 = drugs294[drugs294["hour"] >= 12]
rates294 = {}
for drug294 in kernels294:
    sub294 = drugs294[drugs294["event"] == drug294]
    rates294[drug294] = sub294.groupby(["stay_id", "ihour"])["rate"].sum().to_dict()
print("Infusion-hour fractional parts:",
      np.round((drugs294["hour"] % 1).quantile([0, .25, .5, .75, 1]).values, 4))

editcol294 = next((c for c in ("edit", "intervention", "dose_edit") if c in roll294), None)
hourcol294 = next((c for c in ("hour", "stay_hour", "icu_hour") if c in roll294), None)
if editcol294 and hourcol294 and "stay_id" in roll294:
    target_summary294 = {}
    for label294, drug294, scale294 in [
        ("insulin_x2", "insulin", 2.0), ("insulin_half", "insulin", .5),
        ("ne_x2", "norepinephrine", 2.0), ("ne_half", "norepinephrine", .5)
    ]:
        rows294 = roll294[roll294[editcol294] == label294].copy()
        if "reached" in rows294:
            rows294 = rows294[rows294["reached"].astype(str).str.lower().isin(["true", "1", "1.0"])]
        rows294 = rows294[(rows294[hourcol294] >= 13) & (rows294[hourcol294] <= 24)]
        rows294 = rows294.drop_duplicates(["stay_id", hourcol294])
        vals294 = []
        for sid294, hour294 in rows294[["stay_id", hourcol294]].itertuples(index=False, name=None):
            t294 = int(np.floor(hour294 + 1e-7))
            vals294.append((scale294-1) * sum(
                k294 * rates294[drug294].get((sid294, t294-l294), 0.0)
                for l294, k294 in enumerate(kernels294[drug294], start=1)
            ))
        target_summary294[label294] = {
            "six_lag_estimate": float(np.mean(vals294)) if vals294 else None,
            "n_rows": len(vals294)
        }
    print("RECORD-DERIVED SIX-LAG TARGETS (conditional-exogeneity and tail limitations):",
          target_summary294)
else:
    print("Need to adapt rollout-table column mapping before estimating matched targets.")

print("Projection cache names:",
      " ".join(n for n in globals() if n.endswith("286"))[:1600])
print("Next baseline factual request:")
print({"type": "factual", "model": "both", "split": "tuning", "stays": validation_ids293})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 294\n"
            "Forward34 current [-3,.4] candidate worsens independent factual NLL by "
            f"{paired294['delta_nll'].mean():+.8f}, above .005 on this batch. "
            "Saved in checkpoint292; removed b_in edit to isolate safe236 baseline. "
            "Computed matched six-lag record-derived targets where rollout columns permit; "
            "these retain conditional-exogeneity and omitted-tail uncertainty. "
            "Counts: 294 Python runs, 34 forwards.\n")
```
