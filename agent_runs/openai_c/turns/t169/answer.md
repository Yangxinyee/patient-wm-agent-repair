Across 200 stays, the current edit increased factual NLL by **0.00238 nats**; the stay-clustered one-sided 95% upper bound is **0.00438**, below the limit. That supports keeping it. The rollout summary hit a Pandas naming error without changing any edits; I’ll fix that and select the final independent cohort.

```python
from pathlib import Path
import numpy as np
import pandas as pd

rollout47 = {}
for model in ("original", "edited"):
    df = pd.read_csv(f"/work/fwd_47/dose_edit_{model}.csv")
    reached = df.loc[df["reached"] == True]
    by_stay = reached.groupby(["edit", "stay_id"])["shift"].mean()
    rollout47[model] = {}
    for edit, values in by_stay.groupby(level="edit"):
        rollout47[model][edit] = {
            "mean": float(values.mean()),
            "stay_se": float(values.std(ddof=1) / np.sqrt(len(values))),
            "stays": len(values)
        }
print("GAIN 1.30 ROLLOUT:", rollout47)

for name in ("target_train", "target_probe"):
    if name in globals():
        print(name, repr(globals()[name])[:1200])

tuning_stays = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist())
previous_check_stays = set()
for folder in Path("/work").glob("fwd_*"):
    for filename in ("factual_original.csv", "dose_edit_original.csv"):
        path = folder / filename
        if path.exists():
            previous_check_stays.update(pd.read_csv(
                path, usecols=["stay_id"])["stay_id"].unique().tolist())
final_validation_ids = [
    int(s) for s in tuning_stays if s not in previous_check_stays
][:100]
print("PREVIOUSLY CHECKED TUNING STAYS:",
      len(set(tuning_stays) & previous_check_stays))
print("FINAL VALIDATION IDS:", final_validation_ids)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python runs 116–117: gain-1.30 final audit\n"
        f"- Paired factual audit on disjoint cohorts: {audit_130}\n"
        "- Overall NLL increase and its clustered one-sided 95% upper bound"
        " are below +0.005; this is evidence, not a guarantee for evaluation.\n"
        f"- Forward 47 readouts: {rollout47}\n"
        f"- Previously unchecked tuning stays available: {len(final_validation_ids)}.\n"
        "- Run 116 stopped on a Pandas method/column naming collision;"
        " corrected without altering any operations.\n"
        "- No edits changed. Forward passes used: 48; two remain.\n"
    )
```
