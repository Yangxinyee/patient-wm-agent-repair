The stronger trial worsened factual NLL by about 0.0081 on the difficult cohort, so it is not accepted. I’ll check the full tuning-set coverage before deciding how far to reduce it: the acceptance criterion is the overall mean, not the worst cohort. A pooled increase with adequate margin below 0.005 would support retaining some strengthening.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

def paired_factual_resume(k):
    folder = Path(f"/work/fwd_{k}")
    a = pd.read_csv(folder / "factual_original.csv")
    b = pd.read_csv(folder / "factual_edited.csv")
    keys = ["stay_id", "hour", "event", "observed"]
    for frame in (a, b):
        frame["_occurrence"] = frame.groupby(keys, dropna=False).cumcount()
    keys += ["_occurrence"]
    merged = a.merge(b, on=keys, suffixes=("_original", "_edited"),
                     validate="one_to_one")
    assert len(merged) == len(a) == len(b)
    merged["delta"] = merged["nll_edited"] - merged["nll_original"]
    stay = merged.groupby("stay_id")["delta"].agg(["sum", "count"])
    mean = merged["delta"].mean()
    influence = stay["sum"] - mean * stay["count"]
    se = np.sqrt(len(stay) / (len(stay) - 1) *
                 np.square(influence).sum()) / stay["count"].sum()
    return merged, {"forward": k, "stays": len(stay), "readings": len(merged),
                    "min_id": int(stay.index.min()),
                    "max_id": int(stay.index.max()),
                    "delta": float(mean), "cluster_se": float(se)}

factual_history221 = []
for folder in sorted(Path("/work").glob("fwd_*"),
                     key=lambda p: int(p.name.split("_")[-1])):
    if ((folder / "factual_original.csv").exists() and
        (folder / "factual_edited.csv").exists()):
        _, summary = paired_factual_resume(int(folder.name.split("_")[-1]))
        factual_history221.append(summary)
print("PAIRED FACTUAL HISTORY")
print(pd.DataFrame(factual_history221).to_string(index=False))

trial_factual29, trial_summary29 = paired_factual_resume(29)
all_tuning_ids221 = sorted(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique())
tested_trial_ids221 = set(trial_factual29["stay_id"].unique())
remaining_trial_ids221 = [
    int(s) for s in all_tuning_ids221 if s not in tested_trial_ids221]
remaining_trial_batches221 = [
    remaining_trial_ids221[i:i+100]
    for i in range(0, len(remaining_trial_ids221), 100)]
print("\nTUNING COVERAGE:", len(all_tuning_ids221), "total;",
      len(tested_trial_ids221), "tested with this trial")
for i, ids in enumerate(remaining_trial_batches221, 1):
    print(f"REMAINING FACTUAL REQUEST {i}")
    print(json.dumps({"type": "factual", "model": "both",
                      "split": "tuning", "stays": ids}))

print("\nCACHED RECORD-EFFECT EVIDENCE")
for name in ("drug_effects", "iv_persistence_table", "target_summary217"):
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__)
        if isinstance(obj, pd.DataFrame):
            print(obj.head(12).to_string(index=False))
        else:
            print(repr(obj)[:2200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward29 / Run221: stronger trial fails difficult cohort\n"
            f"Paired difficult-cohort factual summary: {trial_summary29}. "
            "candidate_B219 is not accepted. Recovering complete tuning "
            "coverage and prior evidence before choosing gain reduction; "
            "weights unchanged and fallback_B219 remains exact.\n")
```
