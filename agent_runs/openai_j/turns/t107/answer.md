Changing the register layout reduced the prediction error, but not enough to establish an exact reconstruction. I’ll test combinations of presence flags and inspect the rejected GPU edit by sensor. If its error is mainly a stable offset tied to sensor and last value, a calibrated counterterm may help; large unpredictable variation would make that approach unsafe.

```python
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

print("=== REGISTER FLAG COMBINATIONS ===")
flag_results94 = []
with torch.no_grad():
    for value_col94 in (2, 6):
        flag_cols94 = [j for j in range(2, 8) if j != value_col94]
        for flags94 in itertools.product((0, 1), repeat=len(flag_cols94)):
            numbers94 = torch.zeros_like(old_numbers93)
            numbers94[:, 0] = has93
            numbers94[:, 1] = age93
            numbers94[:, value_col94] = value93
            active94 = []
            for j94, on94 in zip(flag_cols94, flags94):
                if on94:
                    numbers94[:, j94] = has93
                    active94.append(j94)
            inputs94 = fixed_inputs93 + F.linear(numbers94, numeric_W93, numeric_b93)
            H94 = F.gelu(F.linear(inputs94, wt("attr_heads.value_bin.0.weight"),
                                 wt("attr_heads.value_bin.0.bias")))
            P94 = (F.linear(H94, wt("attr_heads.value_bin.2.weight"),
                            wt("attr_heads.value_bin.2.bias"))+transition93).softmax(-1)
            mean94 = (P94*centres93).sum(-1).numpy()
            err94 = (mean94-target93)/scales93
            nll94 = -P94[torch.arange(len(P94)), target_bins93].clamp_min(1e-30).log()
            flag_results94.append({
                "value_col": value_col94, "presence_cols": str(active94),
                "rmse": np.sqrt(np.mean(err94**2)), "bias": err94.mean(),
                "nll": float(nll94.mean())})
flag_table94 = pd.DataFrame(flag_results94).sort_values("rmse")
print(flag_table94.head(12).to_string(index=False))

print("\n=== REJECTED GPU CANDIDATE ===")
orig94 = pd.read_csv("/work/fwd_12/factual_original.csv")
edit94 = pd.read_csv("/work/fwd_12/factual_edited.csv")
cmp94 = orig94.merge(edit94, on=["stay_id", "hour", "event", "observed"],
                     suffixes=("_original", "_edited"), validate="one_to_one")
cmp94["dnll"] = cmp94.nll_edited-cmp94.nll_original
cmp94["dmean"] = cmp94.predicted_mean_edited-cmp94.predicted_mean_original
print(cmp94.groupby("event").agg(
    n=("dnll", "size"), dnll=("dnll", "mean"),
    mean_shift=("dmean", "mean"), shift_sd=("dmean", "std"),
    shift_min=("dmean", "min"), shift_max=("dmean", "max")).to_string())
cmp94["period"] = pd.cut(cmp94.hour, [-10, 0, 6, 12, 24, 100])
print(cmp94.groupby(["event", "period"], observed=True).agg(
    n=("dnll", "size"), dnll=("dnll", "mean"),
    mean_shift=("dmean", "mean"), shift_sd=("dmean", "std")).to_string())

print("\n=== BIN TABLE STRUCTURE ===")
print("VOCAB keys:", list(VOCAB))
print("centre example:", {k: v for k, v in VOCAB.items()
                          if "cent" in k.lower()})
print("ACTIVE EDITS:", edited())
```