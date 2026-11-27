The raw state retains some insulin-innovation information, but little reliable norepinephrine-innovation information. I’ll now compute provisional rollout targets from the record-based impulse estimates and test a reversible action-encoder amplification. It is only a probe: stronger dose responses would support it, while worse factual loss or unchanged responses would refute it.

```python
import numpy as np
import pandas as pd
import torch
import json

print("=== RECORD-BASED ROLLOUT TARGETS ===")
target_rows29 = []
for edit29, drug29, multiplier29 in (
    ("insulin_x2", "insulin", 2.0),
    ("insulin_half", "insulin", 0.5),
    ("ne_x2", "norepinephrine", 2.0),
    ("ne_half", "norepinephrine", 0.5),
):
    coeff29 = np.asarray(irf_coefficients[drug29], dtype=float)
    if drug29 == "insulin":
        ratio29 = np.clip(np.median(coeff29[-3:-1] / coeff29[-4:-2]), 0.0, 0.95)
        extended29 = np.r_[coeff29, coeff29[-1] * ratio29 ** np.arange(1, 17)]
    else:
        extended29 = np.r_[coeff29, np.zeros(16)]

    subset29 = dose_file28[
        (dose_file28.edit == edit29) & dose_file28.reached.astype(bool)
    ].copy()
    estimates29 = []
    truncated29 = []
    for row29 in subset29.itertuples():
        h29 = int(round(row29.hour))
        total29 = 0.0
        short29 = 0.0
        for action_hour29 in range(12, h29):
            key29 = (int(row29.stay_id), action_hour29)
            rate29 = float(tpanel.loc[key29, drug29]) if key29 in tpanel.index else 0.0
            lag29 = h29 - action_hour29 - 1
            if 0 <= lag29 < len(extended29):
                total29 += (multiplier29 - 1.0) * rate29 * extended29[lag29]
            if 0 <= lag29 < len(coeff29):
                short29 += (multiplier29 - 1.0) * rate29 * coeff29[lag29]
        estimates29.append(total29)
        truncated29.append(short29)
    subset29["record_estimate"] = estimates29
    subset29["truncated_estimate"] = truncated29
    target_rows29.append({
        "edit": edit29,
        "exposed_stays": subset29.stay_id.nunique(),
        "original_prediction": subset29.groupby("stay_id")["shift"].mean().mean(),
        "record_estimate_8_lags": subset29.groupby("stay_id")["truncated_estimate"].mean().mean(),
        "record_estimate_with_tail": subset29.groupby("stay_id")["record_estimate"].mean().mean(),
    })

targets29 = pd.DataFrame(target_rows29)
print(targets29.round(4).to_string(index=False))
print("These are estimated targets, not observed intervention truth.")

print("\n=== REVERSIBLE ACTION-SCALE PROBE ===")
assert len(edited()) == 0, "Unexpected edits: inspect before replacing them."
action_gain29 = 2.0
for name29 in ("encode_action.norm.weight", "encode_action.norm.bias"):
    original29 = torch.as_tensor(load_original(name29), dtype=torch.float32)
    save_edit(name29, original29 * action_gain29)
print("Edited:", edited())

probe_stays29 = [int(s) for s in np.asarray(tune_dev).reshape(-1)]
request29 = {
    "type": "factual", "model": "both", "split": "tuning",
    "stays": probe_stays29
}
print("\n=== NEXT FACTUAL REQUEST ===")
print(json.dumps(request29))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 29; forward 4 — first reversible edit\n")
    f.write("- Estimated rollout targets by convolving recorded post-hour-12 rates with controlled impulse estimates; these are not intervention truth.\n")
    f.write(targets29.round(4).to_string(index=False) + "\n")
    f.write("- Probe only: multiplied encode_action.norm.weight and bias by 2, scaling action bundle vectors while leaving action-free bundles unchanged.\n")
    f.write("- Must verify factual value-NLL <= original +0.005 and inspect paired dose responses; revert if unsuitable.\n")
```