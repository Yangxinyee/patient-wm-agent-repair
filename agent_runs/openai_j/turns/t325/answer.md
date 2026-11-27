The final batch’s NLL increase is 0.00441 nats. Together with the other batches, this suggests the candidate passes on average, but the margin may be narrow. I’ll compute the exact pooled loss and stay-cluster uncertainty. A comfortable margin supports retention; weak safety evidence favors blending toward the validated fallback.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

fact_dirs280 = ["/work/fwd_37", "/work/fwd_38", "/work/fwd_34", "/work/fwd_39"]
delta_frames280 = []
for d280 in fact_dirs280:
    o280 = pd.read_csv(d280 + "/factual_original.csv")
    e280 = pd.read_csv(d280 + "/factual_edited.csv")
    if not delta_frames280:
        print("FACTUAL CSV COLUMNS:", o280.columns.tolist())
    keys280 = [c for c in ("stay_id", "hour", "event") if c in o280.columns]
    assert len(o280) == len(e280)
    assert o280[keys280].equals(e280[keys280]), "Row alignment must be checked"
    nllcols280 = [c for c in o280.columns if "nll" in c.lower()]
    nllcol280 = next((c for c in ("nll", "value_nll", "fine_bin_nll") if c in nllcols280), nllcols280[0])
    q280 = o280[keys280].copy()
    q280["delta_nll"] = e280[nllcol280].to_numpy() - o280[nllcol280].to_numpy()
    q280["original_nll"] = o280[nllcol280].to_numpy()
    q280["edited_nll"] = e280[nllcol280].to_numpy()
    delta_frames280.append(q280)
    print(d280, "stays", q280.stay_id.min(), q280.stay_id.max(),
          "readings", len(q280), "delta", q280.delta_nll.mean())

pooled280 = pd.concat(delta_frames280, ignore_index=True)
assert pooled280.stay_id.nunique() == 400
print("POOLED:", pooled280[["original_nll", "edited_nll", "delta_nll"]].mean().to_dict())
cluster280 = pooled280.groupby("stay_id").delta_nll.agg(["sum", "count"])
rng280 = np.random.default_rng(280)
ix280 = rng280.integers(0, len(cluster280), size=(10000, len(cluster280)))
boot280 = cluster280["sum"].to_numpy()[ix280].sum(axis=1) / cluster280["count"].to_numpy()[ix280].sum(axis=1)
print("STAY-BOOTSTRAP: SE", boot280.std(), "95% interval", np.quantile(boot280, [0.025, 0.975]))
if "event" in pooled280:
    print("BY EVENT:", pooled280.groupby("event").delta_nll.agg(["mean", "count"]).to_string())

print("=== SAVED VARIANTS ===")
for fn280 in ("validated_safe_backup268.npz", "calibrated_candidate268.npz",
              "pre_action_extrap_backup277.npz"):
    with np.load("/work/" + fn280) as z280:
        print(fn280, z280.files)
        print({k: float(np.max(np.abs(load(k) - z280[k])))
               for k in z280.files if k in SHAPES})

print("=== CANDIDATE HELPERS ===")
names280 = sorted(k for k in globals()
                  if any(s in k.lower() for s in ("candidate", "calib", "strength", "likelihood", "restore"))
                  and not k.endswith("280"))
print(names280[:90])
print("=== CALIBRATION FRONTIER, FIRST 8 ROWS ===")
print(pd.read_csv("/work/calibration_strength_frontier265.csv").head(8).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 280: pooled calibrated-candidate factual audit\n"
            f"- Current four-tensor candidate pooled tuning dNLL: {pooled280.delta_nll.mean():.8f}.\n"
            f"- Stay-bootstrap SE: {boot280.std():.8f}; 95% interval: {np.quantile(boot280, [0.025, 0.975]).tolist()}.\n"
            "- GPU batches: fwd_37, fwd_38, fwd_34, fwd_39, covering all 400 tuning stays.\n"
            "- No weights changed. Counts: 280 Python runs, 38 forward passes.\n")
```