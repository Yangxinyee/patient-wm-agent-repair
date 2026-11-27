The final cohort adds about 0.00206, leaving the stronger trial’s full-set increase slightly above 0.005. I’ll compute the exact pooled loss and its uncertainty, then measure this trial’s dose responses before reducing it. That will show whether the extra gains buy useful causal accuracy or mostly damage factual fit.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

def pooled_factual_resume(passes):
    frames = [paired_factual_resume(k)[0].assign(forward=k)
              for k in passes]
    pooled = pd.concat(frames, ignore_index=True)
    assert not pooled.duplicated(
        ["stay_id", "hour", "event", "observed", "_occurrence"]).any()
    stay = pooled.groupby("stay_id")["delta"].agg(["sum", "count"])
    mean = pooled["delta"].mean()
    influence = stay["sum"] - mean * stay["count"]
    se = np.sqrt(len(stay) / (len(stay) - 1) *
                 np.square(influence).sum()) / stay["count"].sum()
    summary = {
        "stays": len(stay), "readings": len(pooled),
        "original": float(pooled["nll_original"].mean()),
        "edited": float(pooled["nll_edited"].mean()),
        "delta": float(mean), "cluster_se": float(se),
        "upper_95_one_sided": float(mean + 1.645 * se)}
    return pooled, summary

strong_factual222, strong_summary222 = pooled_factual_resume([29, 30, 31, 32])
assert set(strong_factual222["stay_id"]) == set(all_tuning_ids221)
print("COMPLETE STRONG-TRIAL FACTUAL SUMMARY")
print(json.dumps(strong_summary222, indent=2))
print("\nBY EVENT")
print(strong_factual222.groupby("event").agg(
    readings=("delta", "size"), delta=("delta", "mean"),
    original=("nll_original", "mean"),
    edited=("nll_edited", "mean")).to_string())

dose28_resume = pd.read_csv("/work/fwd_28/dose_edit_edited.csv")
print("\nDOSE FILE FORMAT:", dose28_resume.columns.tolist())
strong_dose_ids222 = [int(s) for s in dose28_resume["stay_id"].drop_duplicates()]
assert len(strong_dose_ids222) <= 100
print("MATCHED STRONG-TRIAL DOSE REQUEST")
print(json.dumps({"type": "dose_edit", "model": "both", "split": "tuning",
                  "stays": strong_dose_ids222}))

print("\nEARLY RECORD-EFFECT NOTEBOOK EVIDENCE")
notes_text222 = Path("/work/notes.md").read_text()
lines222 = notes_text222.splitlines()
matches222 = [
    i for i, line in enumerate(lines222)
    if any(term in line.lower() for term in
           ("discontinu", "instrument", "identified", "identification",
            "drug slope", "causal slope", "persistence estimate"))]
for i in matches222[:12]:
    print("\n".join(lines222[max(0, i-1):min(len(lines222), i+3)]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Forwards29–32 / Run222: complete stronger-trial factual failure\n"
            f"All 400 tuning stays: {strong_summary222}. "
            "The 1.20 insulin / 1.35 NE trial is not accepted. "
            "Next measure its matched untouched-cohort dose responses to "
            "inform a reduced-NE candidate; no weight changes in this run.\n")
```
