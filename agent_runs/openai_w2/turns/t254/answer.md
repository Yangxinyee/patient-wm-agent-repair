On the untouched cohort, factual NLL increased by only about 0.00025, giving a comfortable margin there. I’ll now compute exact pooled factual differences and align the record-derived targets with the rollout’s measured hours. Large discrepancies between those targets and rollout shifts would refute the current calibration even if factual fit passes.

```python
import numpy as np
import pandas as pd
from pathlib import Path

def paired_factual217(k):
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    keys = ["stay_id", "hour", "event", "observed"]
    assert len(o) == len(e) and o[keys].equals(e[keys])
    z = o[keys].copy()
    z["original_nll"] = o.nll.to_numpy()
    z["edited_nll"] = e.nll.to_numpy()
    z["delta_nll"] = z.edited_nll - z.original_nll
    z["cohort"] = k
    return z

def cluster_summary217(z):
    mu = z.delta_nll.mean()
    g = z.groupby("stay_id").delta_nll.agg(["sum", "size"])
    se = np.sqrt(len(g)/(len(g)-1) *
                 np.square(g["sum"] - mu*g["size"]).sum()) / len(z)
    return {"readings": len(z), "stays": len(g),
            "original": z.original_nll.mean(),
            "edited": z.edited_nll.mean(), "delta": mu, "cluster_se": se}

factual_current217 = pd.concat(
    [paired_factual217(26), paired_factual217(27)], ignore_index=True
)
print("CURRENT CANDIDATE FACTUAL")
for k217 in [26, 27]:
    print(k217, cluster_summary217(factual_current217.query("cohort == @k217")))
print("POOLED", cluster_summary217(factual_current217))
print("BY EVENT", factual_current217.groupby("event").delta_nll.mean().to_dict())

record_targets217 = record_impulse_targets(
    tune_panel, sorted(int(x) for x in tuning_ids216),
    -1.8, float(latent_ins158["parameters"]["rho"]),
    3.6, float(latent_ne157["parameters"]["rho"])
)

def target_summary217(z):
    z = z.loc[z.eligible.astype(bool)]
    return pd.DataFrame({
        "pooled_hour_target": z.groupby("edit").target.mean(),
        "equal_stay_target": z.groupby(["edit", "stay_id"]).target.mean().groupby("edit").mean(),
        "eligible_hours": z.groupby("edit").size(),
        "eligible_stays": z.groupby("edit").stay_id.nunique()
    })

print("\nASSUMPTION-DEPENDENT RECORD TARGETS: ALL TUNING")
print(target_summary217(record_targets217).round(5).to_string())
print("\nSAME TARGETS: UNTOUCHED COHORT")
print(target_summary217(record_targets217.loc[
    record_targets217.stay_id.isin(unseen_tuning216)
]).round(5).to_string())

print("\nPRIOR VALIDATED ROLLOUT, MATCHED HOURS")
for m217 in ["original", "edited"]:
    path217 = Path(f"/work/fwd_21/dose_edit_{m217}.csv")
    if not path217.exists():
        continue
    d217 = pd.read_csv(path217)
    print(m217, "columns", list(d217.columns))
    j217 = d217.merge(record_targets217, on=["stay_id", "hour", "edit"],
                     how="left", validate="many_to_one")
    assert j217.target.notna().all()
    j217 = j217.loc[j217.reached.astype(bool)]
    print(j217.groupby("edit")[["shift", "target"]].mean().round(5).to_string())
    print("equal-stay",
          j217.groupby(["edit", "stay_id"])[["shift", "target"]].mean()
              .groupby("edit").mean().round(5).to_dict())

with open("/work/notes.md", "a") as f217:
    f217.write(
        "\n## Forward27 / run217: untouched factual validation and target alignment\n"
        f"Current candidate factual summaries: {cluster_summary217(factual_current217)}. "
        "Record targets use b_insulin=-1.8, rho=0.9696261, b_NE=3.6, "
        "rho_NE=0.8471704; these remain identification/model assumptions, not simulator truth. "
        "Audited pooled-hour versus equal-stay weighting and matched prior rollout hours. "
        "No weight changes.\n"
    )
```
