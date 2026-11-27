The reversal passes held-out factual fit, but only narrowly: +0.00451 nats. I’ll check whether its asymmetric effects come from biased rollout baselines or bin saturation. Baselines near the bin limits would support saturation; otherwise the response dynamics need further repair.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import re

print("DATA README")
print(Path("/data/README.md").read_text())

rollout_tables204 = {}
for label, folder in [
    ("original", "fwd_18"),
    ("ne_plus_half", "fwd_17"),
    ("ne_minus_half", "fwd_18")
]:
    model = "original" if label == "original" else "edited"
    tab = pd.read_csv(f"/work/{folder}/dose_edit_{model}.csv")
    reached = tab["reached"].astype(str).str.lower().isin(["true", "1", "1.0"])
    tab = tab.loc[reached].copy()
    rollout_tables204[label] = tab
    print("\nROLLOUT BASELINES AND SHIFTS", label)
    for edit, a in tab.groupby("edit"):
        print(edit, {
            "shift": float(a["shift"].mean()),
            "equal_stay_shift": float(a.groupby("stay_id")["shift"].mean().mean()),
            "baseline_mean": float(a["factual"].mean()),
            "baseline_quantiles": a["factual"].quantile([.05, .5, .95]).round(3).tolist(),
            "baseline_by_hour": a.groupby("hour")["factual"].mean().round(2).to_dict()
        })

print("\nVOCAB CODES", VOCAB["codes"])
for key in ["value_bin_centres", "normalisation", "normalization"]:
    if key in VOCAB:
        obj = VOCAB[key]
        if isinstance(obj, dict):
            print(key, {k: v for k, v in obj.items()
                        if str(k) in ["glucose_finger", "map_arterial", "0", "2"]})
        else:
            print(key, np.asarray(obj).shape)

tuning204 = pd.read_csv("/data/tuning.csv")
print("\nTUNING COLUMNS", tuning204.columns.tolist())
print(tuning204.head(3).to_string(index=False))

print("\nNONLINEAR RECORD ESTIMATES")
if "quadratic137" in globals():
    print("quadratic137", small_result202(quadratic137))
print("readout_estimates137",
      {drug: {k: (float(v[0]), np.asarray(v[1]).tolist()) for k, v in obj.items()}
       for drug, obj in readout_estimates137.items()})

notes204 = Path("/work/notes.md").read_text()
for run in [136, 137, 170]:
    match = re.search(rf"(?m)^## Run {run}\b.*?(?=\n## |\Z)", notes204, re.S)
    if match:
        print(f"\nRECORD EVIDENCE RUN {run}", match.group(0)[:1300])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 204\nGPU reversed-NE candidate (scale -.5) passed factual on 98 tuning stays: 1.59803 versus original 1.59352, dNLL +.00451. Paired rollout row means: insulin x2 -8.493, half +12.367; NE x2 +1.345, half -4.699. This confirms rollout orientation reversal but not magnitude accuracy. Inspecting free baseline drift/bin saturation and nonlinear record evidence; no weights changed.\n")
```