The untouched cohort’s factual increase is **0.00466 nats**, still below the limit but closer to it. I’ll pool all three current-candidate cohorts before committing the last pass. If the pooled estimate retains a reasonable safety margin, I’ll use that pass to measure all four rollout effects independently.

```python
import json
from pathlib import Path
import pandas as pd

audit_130_final, factual_130_final_rows = factual_audit([46, 48, 49])
print("GAIN 1.30 FINAL FACTUAL AUDIT:", audit_130_final)

print("\nFORWARD 47 TOOL SUMMARY:")
print(json.dumps(json.loads(
    Path("/work/fwd_47/summary.json").read_text()), indent=2))

print("\nFORWARD 47 WEIGHTING CHECK:")
for model in ("original", "edited"):
    df = pd.read_csv(f"/work/fwd_47/dose_edit_{model}.csv")
    reached = df.loc[df["reached"] == True]
    hourly = reached.groupby("edit")["shift"].mean().to_dict()
    staywise = (reached.groupby(["edit", "stay_id"])["shift"].mean()
                .groupby(level="edit").mean().to_dict())
    print(model, {"reached_hour_mean": hourly,
                  "equal_stay_mean": staywise})

print("\nCURRENT OPERATIONS:", ops())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python run 118 / forward 49: untouched factual cohort\n"
        "- New 100-stay factual NLL: original 1.64249, edited 1.64715;"
        " increase +0.00466 nats per reading.\n"
        f"- Pooled exact-candidate audit: {audit_130_final}\n"
        "- Audited tool-summary versus within-stay weighting before"
        " reporting final rollout expectations.\n"
        "- No edits changed. Forward passes used: 49; one remains.\n"
    )
```
