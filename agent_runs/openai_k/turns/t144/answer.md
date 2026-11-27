The channels respond mostly to the correct drug, but their scales are not yet interpretable as physical rates, and norepinephrine is asymmetric under doubling versus halving. The approximate record-based targets are much larger than the original predictions: roughly −11/+5.5 for insulin and +4.2/−2.1 for norepinephrine. I’ll verify the rate units and inspect the failed candidate’s actual factual prediction changes before selecting the next correction route.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=== RAW PHYSICAL RATES VERSUS RETAINED TARGETS ===")
r133_physical = np.zeros((len(rr33_stays), 2), dtype=np.float32)
r133_drug_order = ["insulin", "norepinephrine"]
for r133_i in range(len(rr33_stays)):
    r133_key = state_keys_audit[r133_i]
    try:
        r133_bundle = bundle_records_audit[r133_key]
    except TypeError:
        r133_bundle = bundle_records_audit[tuple(r133_key)]
    for r133_row, r133_kind in r133_bundle:
        if int(r133_kind) == 0 and r133_row.event in r133_drug_order:
            r133_rate = float(r133_row.rate)
            if np.isfinite(r133_rate):
                r133_physical[r133_i, r133_drug_order.index(
                    r133_row.event)] += r133_rate

for r133_d, r133_drug in enumerate(r133_drug_order):
    r133_m = (r133_physical[:, r133_d] > 0) & (rtrue66[:, r133_d] > 0)
    print(r133_drug,
          "physical/retained median ratio:",
          float(np.median(r133_physical[r133_m, r133_d] /
                          rtrue66[r133_m, r133_d])),
          "rate66 versus physical MAE:",
          float(np.mean(np.abs(rate66[:, r133_d] -
                               r133_physical[:, r133_d]))))
    r133_sample = np.flatnonzero(r133_m)[:5]
    print(np.c_[r133_sample, r133_physical[r133_sample, r133_d],
                rtrue66[r133_sample, r133_d], rate66[r133_sample, r133_d]])

print("\n=== CHANNEL RESPONSE IN PHYSICAL UNITS ===")
for r133_d, r133_drug in enumerate(r133_drug_order):
    r133_m = r133_physical[r132_ix, r133_d] > 0
    for r133_factor in [0.0, 0.5, 2.0, 3.0]:
        r133_u = rr35_scaled_action_encoding(r132_ix, r133_drug, r133_factor)
        r133_dz = (r132_channels(r132_ix, r133_u) - r132_z)[r133_m]
        r133_dr = ((r133_factor - 1) *
                   r133_physical[r132_ix, r133_d][r133_m])
        r133_slopes = r133_dr @ r133_dz / (r133_dr @ r133_dr)
        print(r133_drug, r133_factor, np.round(r133_slopes, 4))

print("\n=== FAILED CANDIDATE'S FACTUAL MEAN CHANGES ===")
r133_orig_path = Path("/work/fwd_5/factual_original.csv")
r133_edit_path = Path("/work/fwd_5/factual_edited.csv")
if r133_orig_path.exists() and r133_edit_path.exists():
    r133_fo = pd.read_csv(r133_orig_path)
    r133_fe = pd.read_csv(r133_edit_path)
    print("columns:", list(r133_fo))
    r133_mean_cols = [c for c in r133_fo if "mean" in c.lower()]
    if r133_mean_cols and len(r133_fo) == len(r133_fe):
        r133_mean_col = r133_mean_cols[0]
        r133_delta_mean = r133_fe[r133_mean_col] - r133_fo[r133_mean_col]
        print("overall mean/SD/max absolute:",
              float(r133_delta_mean.mean()), float(r133_delta_mean.std()),
              float(r133_delta_mean.abs().max()))
        r133_event_col = next(
            (c for c in ["event", "code", "event_name"] if c in r133_fo), None)
        if r133_event_col:
            r133_summary = pd.DataFrame({
                "event": r133_fo[r133_event_col],
                "delta": r133_delta_mean})
            print(r133_summary.groupby("event").delta.agg(
                ["mean", "std", "min", "max"]).round(5).to_string())

print("\n=== AVAILABLE PRIOR-HEAD CALIBRATION ARRAYS ===")
for r133_n in ["J65", "D65_candidates", "direction", "coef88",
               "dr66", "delta88", "target88"]:
    r133_obj = globals().get(r133_n)
    if isinstance(r133_obj, dict):
        print(r133_n, {
            str(k): tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            for k, v in r133_obj.items()})
    elif hasattr(r133_obj, "shape"):
        print(r133_n, tuple(r133_obj.shape))
print("target88:", target88)

with open("/work/notes.md", "a") as r133_file:
    r133_file.write(
        "\n## Run 133: physical units and official factual perturbation audit\n"
        "- Approximate tuning targets: insulin x2 -10.37 to -11.39, half +5.18 to +5.70; NE x2 +4.21 to +4.25, half -2.10 to -2.13.\n"
        "- Auditing raw infusion rates against retained decoder targets and measuring official factual mean changes of the rejected candidate.\n"
        "- Python runs: 133; forward passes: 6; no submitted edits.\n"
    )
```