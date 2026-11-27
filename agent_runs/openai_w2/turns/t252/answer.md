The current edit is rank two, and its exact fallback is available. That confirms the intended construction, but not its causal accuracy. Next I’ll recover the estimated drug effects, persistence assumptions, and untouched validation stays; agreement across record analyses would support the rollout targets, while disagreement would call for more conservative calibration.

```python
import pandas as pd
import numpy as np
import torch
from pathlib import Path

def compact216(x, depth=0):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, pd.DataFrame):
        num = x.select_dtypes(include=[np.number])
        return {
            "shape": x.shape,
            "columns": list(x.columns),
            "numeric_means": num.mean().round(5).to_dict(),
            "head": x.head(2).to_dict("records")
        }
    if isinstance(x, pd.Series):
        return {"shape": x.shape, "head": x.head(8).to_dict()}
    if isinstance(x, np.ndarray):
        if x.size <= 20:
            return x.tolist()
        return {"shape": x.shape, "mean": float(np.nanmean(x)),
                "min": float(np.nanmin(x)), "max": float(np.nanmax(x))}
    if isinstance(x, dict):
        if depth >= 2:
            return {"keys": list(x.keys())[:16]}
        return {str(k): compact216(v, depth+1) for k, v in list(x.items())[:16]}
    if isinstance(x, (tuple, list)):
        if len(x) <= 8:
            return [compact216(v, depth+1) for v in x]
        return {"length": len(x), "first": [compact216(v, depth+1) for v in x[:3]]}
    if isinstance(x, (float, int, bool, str, np.number)) or x is None:
        return x
    return type(x).__name__

for n216 in ["drug_effects", "target_ranges", "rho_iv", "latent_fits",
             "latent_ne157", "latent_ins158", "record_targets", "aligned_targets"]:
    if n216 in globals():
        text216 = repr(compact216(globals()[n216]))
        print(n216, text216[:1200])

print("\nDATA PANEL OBJECTS")
for n216, o216 in list(globals().items()):
    if isinstance(o216, pd.DataFrame) and re.search(r"panel|^p$|train|tune|gold", n216, re.I):
        print(n216, o216.shape, list(o216.columns)[:18])

tuning_rows216 = pd.read_csv("/data/tuning.csv", usecols=["stay_id", "event"])
tuning_ids216 = set(tuning_rows216.stay_id.unique())
usable_tuning216 = set(tuning_rows216.loc[
    tuning_rows216.event.isin(["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]),
    "stay_id"
].unique())

seen_tuning216 = set()
cohorts216 = {}
for path216 in sorted(Path("/work").glob("fwd_*/factual_original.csv"),
                      key=lambda p: int(p.parent.name.split("_")[1])):
    ids216 = set(pd.read_csv(path216, usecols=["stay_id"]).stay_id.unique()) & tuning_ids216
    if ids216:
        cohorts216[path216.parent.name] = sorted(ids216)
        seen_tuning216 |= ids216

unseen_tuning216 = sorted(usable_tuning216 - seen_tuning216)[:100]
print("\nFACTUAL COHORTS", [(k, len(v), min(v), max(v)) for k, v in cohorts216.items()])
print("UNTOUCHED TUNING REQUEST IDS", unseen_tuning216)
print("CURRENT equals candidate_B212",
      np.max(np.abs(current_B215.numpy() - np.asarray(candidate_B212))))

with open("/work/notes.md", "a") as f216:
    f216.write(
        "\n## Run216: effect and validation inventory\n"
        "Current candidate remains the rank-two constrained insulin / pooled NE edit. "
        "Recovered record-effect and persistence objects and identified untouched "
        "tuning stays. No weight changes.\n"
    )
```
