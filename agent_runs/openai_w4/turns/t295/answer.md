The dose fixtures match the panel’s raw rates exactly, so the recovered kernels need no dose rescaling. The rollout file also supplies an explicit eligibility flag. I’ll compute matched targets under both row-weighted and stay-weighted averaging, with bootstrap intervals. This will show whether averaging explains any of the current calibration gap.

```python
import json
import dis
import numpy as np
import pandas as pd
from pathlib import Path

print("LATEST FORWARD SUMMARY")
summary256 = json.loads(
    (latest_dose_path255.parent / "summary.json").read_text())
print(json.dumps(summary256, indent=2)[:2600])

tuning_all256 = pd.read_csv("/data/tuning.csv")
tuning_panel256 = make_panel(tuning_all256)
matched256 = latest_dose255.copy()
target_results256 = {}
edit_specs256 = {
    "insulin_x2": ("insulin", 1.0),
    "insulin_half": ("insulin", -0.5),
    "ne_x2": ("norepinephrine", 1.0),
    "ne_half": ("norepinephrine", -0.5),
}

for edit256, (drug256, delta256) in edit_specs256.items():
    rows256 = matched256.loc[
        (matched256["edit"] == edit256) & matched256["reached"].astype(bool)
    ].copy()
    exposures256 = np.zeros((len(rows256), 12))
    sid256 = rows256["stay_id"].to_numpy()
    hour256 = rows256["hour"].to_numpy()
    rates256 = tuning_panel256[drug256].fillna(0)
    for lag256 in range(12):
        source_hour256 = hour256 - lag256 - 1
        lookup256 = pd.MultiIndex.from_arrays(
            [sid256, source_hour256], names=rates256.index.names)
        exposures256[:, lag256] = (
            rates256.reindex(lookup256).fillna(0).to_numpy()
            * (source_hour256 >= 12) * delta256)
    rows256["record_target"] = exposures256 @ kernels137[drug256]
    row_exposure256 = exposures256.mean(axis=0)
    stay_exposure256 = (
        pd.DataFrame(exposures256).assign(stay_id=sid256)
        .groupby("stay_id").mean().mean(axis=0).to_numpy())
    boot256 = np.asarray(boot_kernels137[drug256]) @ stay_exposure256
    result256 = {
        "hours": len(rows256), "stays": rows256["stay_id"].nunique(),
        "model_row_mean": rows256["shift"].mean(),
        "model_stay_mean": rows256.groupby("stay_id")["shift"].mean().mean(),
        "target_row_mean": rows256["record_target"].mean(),
        "target_stay_mean": rows256.groupby("stay_id")["record_target"].mean().mean(),
        "kernel_bootstrap_95pct": np.quantile(boot256, [.025, .975]).tolist(),
    }
    target_results256[edit256] = result256
    print("\n", edit256, json.dumps(result256))
    matched256.loc[rows256.index, "record_target"] = rows256["record_target"]

print("\nCPU FACTUAL HELPER")
for instruction256 in dis.get_instructions(factual_eval194):
    print(instruction256.opname, instruction256.argrepr)

print("\nINSULIN CANDIDATE ROW DIFFERENCES")
a256 = r244_candidates[(1.0, 26.0)]
b256 = r244_candidates[(1.25, 28.0)]
c256 = r244_candidates[(1.5, 30.0)]
d256 = r244_candidates[(1.5, 34.0)]
for name256 in ("z_post.0.weight", "z_post.0.bias"):
    diff256 = np.asarray(b256[name256]) - np.asarray(a256[name256])
    nz256 = np.flatnonzero(np.abs(diff256).reshape(diff256.shape[0], -1)
                           .max(axis=1) > 1e-8)
    print(name256, "changed rows", nz256.tolist(),
          "max difference", float(np.abs(diff256).max()))
    if name256.endswith("bias"):
        print("bias changes", diff256[nz256],
              "gate-only changes", (np.asarray(d256[name256])
                                    - np.asarray(c256[name256]))[nz256])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 256\n"
            "Training identification dose fixtures match raw panel rates exactly. "
            "Computed exact reached-hour convolution targets and both row/stay "
            "aggregation from latest GPU output. Results: "
            + json.dumps(target_results256) + "\n"
            "Bootstrap intervals quantify kernel estimation only, not identification "
            "or extrapolation bias. No weights changed.\n")
```