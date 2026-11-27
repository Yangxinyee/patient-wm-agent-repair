The candidate reserves two latent coordinates for drug innovations; its scale changes only their token directions. However, the posterior reconstruction has very large weight changes, so I cannot assume its zero-correction version preserves the original rollout. I’ll test that baseline first. If it materially changes dose responses, the correction must be measured against that baseline rather than against the original alone.

```python
from pathlib import Path
import pandas as pd
import numpy as np, torch

for name in list(edited()):
    drop_edit(name)

baseline138 = {
    name: torch.as_tensor(value).detach().clone()
    for name, value in candidates90[0.0].items()
}
for name, value in baseline138.items():
    save_edit(name, value)

print("=== SUBMITTED ZERO-CORRECTION BASELINE ===")
print(edited())
for j in (30, 31):
    print("correction column", j,
          "norm:", float(baseline138["z_in.weight"][:, j].norm()))

path138 = Path("/work/fwd_6/dose_edit_original.csv")
if path138.exists():
    table138 = pd.read_csv(path138)
    print("forward-6 columns:", table138.columns.tolist())
    stay_col138 = next((c for c in ["stay_id", "stay"] if c in table138.columns), None)
    if stay_col138 is not None:
        calibration_ids138 = table138[stay_col138].drop_duplicates().tolist()[:100]
    else:
        calibration_ids138 = pd.read_csv("/data/tuning.csv")["stay_id"].drop_duplicates().tolist()[:100]
else:
    calibration_ids138 = pd.read_csv("/data/tuning.csv")["stay_id"].drop_duplicates().tolist()[:100]

calibration_ids138 = [
    int(x) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
    for x in calibration_ids138
]
print("CALIBRATION STAYS:", calibration_ids138)
print("direction norms:", [
    float(torch.as_tensor(candidates90[1.0]["z_in.weight"])[:, j].norm())
    for j in (30, 31)
])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 138: submit zero-correction baseline for official assay\n"
            "- Candidate scale changes only two reserved drug-token columns.\n"
            "- Submitted candidates90[0.0]; baseline reconstruction needs independent rollout validation.\n"
            "- Python runs: 138; forward passes: 6.\n"
            f"- Calibration stays: {calibration_ids138}\n")
```