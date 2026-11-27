The installed weights exactly match the safety candidate. The recovered estimates support a negative insulin effect on glucose and a positive norepinephrine effect on MAP, but some estimates are unstable and depend on identification assumptions. I’ll now check the validation-group coverage and estimation rationale with compact output. Missing groups or contradictory evidence would weaken the case for submission.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import re

root = Path("/work")
print("COMPLETE ROLLOUT TARGET ESTIMATES:")
print(pd.read_csv(root / "kernel_targets.csv").to_string(index=False))

print("\nRECENT FORWARD COVERAGE:")
for k in range(39, 49):
    folder = root / f"fwd_{k}"
    if not folder.exists():
        continue
    filenames = sorted(p.name for p in folder.iterdir())
    print(folder.name, filenames)
    for filename in ("dose_edit_edited.csv", "factual_edited.csv"):
        p = folder / filename
        if not p.exists():
            continue
        df = pd.read_csv(p)
        idcol = next((c for c in df.columns
                      if c.lower() in ("stay", "stay_id", "stayid")), None)
        if idcol is not None:
            ids = sorted(df[idcol].dropna().unique().tolist())
            print(" ", filename, "rows", len(df), "n_stays", len(ids),
                  "range", ids[:2], ids[-2:])
        if k == 48:
            print("  columns:", list(df.columns))
            print(df.head(2).to_string(index=False))

print("\nFACTUAL AUDIT:")
audit = pd.read_csv(root / "final_factual_audit289.csv")
print("shape", audit.shape, "columns", list(audit.columns))
print(audit.head(5).to_string(index=False)[:1000])

notes = (root / "notes.md").read_text()
print("\nESTIMATION RATIONALE:")
patterns = [
    r"(?im)^.*(?:measurement.error|sensor.disagreement|exclusion|identification).*$",
    r"(?im)^.*(?:kernel.*(?:rho|beta|fit)|(?:rho|beta).*kernel).*$",
    r"(?im)^.*(?:Run 285|safety candidate installed).*$",
]
remaining = 2400
for pattern in patterns:
    hits = list(re.finditer(pattern, notes))
    for m in hits[-2:]:
        excerpt = notes[max(0, m.start()-100):min(len(notes), m.end()+500)]
        excerpt = excerpt[:min(700, remaining)]
        if excerpt:
            print(excerpt)
            remaining -= len(excerpt)

tune_ids_final = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])
                        ["stay_id"].unique().tolist())
print("\nTUNING IDS:", len(tune_ids_final),
      "first", tune_ids_final[:3], "last", tune_ids_final[-3:])
print("FOURTH GROUP EXACT IDS:", tune_ids_final[300:400])

with (root / "notes.md").open("a") as f:
    f.write(
        "\n## Run 294: compact coverage and identification audit\n"
        "- Current four tensors exactly match safety_candidate285.npz.\n"
        "- Recovered DML and sensor-IV estimates support negative insulin/glucose "
        "and positive NE/MAP effects, conditional on their assumptions; "
        "some dynamic-IV estimates are unstable.\n"
        "- Audited forward coverage and exact fourth-group IDs; no weights changed.\n"
        "- Counts: 294 Python runs, 46 forward passes; 6 and 4 remain.\n"
    )
```