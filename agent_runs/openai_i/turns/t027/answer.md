Changing only the register flags produced a modest improvement, not a match. That weakens the hypothesis that a single missing flag caused the discrepancy. I’ll test which column holds the normalized previous value and recover the baseline readouts over exposed stay-hours. A near-exact match would settle the layout; otherwise, edits will need validation directly through the forward tool.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
from pathlib import Path

n22 = 350
base22 = prebase21[:n22]
has22 = has21[:n22]
z22 = z21[:n22]
centres22 = centres21[:n22]
transition22 = transition21[:n22]
reference22 = reference21[:n22]
observed22 = observed_bins21[:n22]
allflags22 = torch.tensor(
    [[float((m >> j) & 1) for j in range(8)] for m in range(256)]
)
flagvectors22 = allflags22 @ effect21
grid22 = []
for age_name in ("zero", "hours_log_div10", "seconds_log_div10"):
    age_tensor = torch.tensor(age_options21[age_name][:n22], dtype=torch.float32)
    for value_column in range(8):
        pre = (base22[None, :, :]
               + z22[None, :, None] * effect21[value_column]
               + age_tensor[None, :, None] * effect21[1]
               + has22[None, :, None] * flagvectors22[:, None, :])
        probabilities = torch.softmax(
            F.gelu(pre) @ w221.T + b221 + transition22[None, :, :], -1
        )
        errors = (probabilities * centres22[None, :, :]).sum(-1) - reference22
        nlls = -torch.log(probabilities.gather(
            2, observed22[None, :, None].expand(256, -1, 1)
        ).squeeze(-1).clamp_min(1e-30)).mean(1)
        rms = errors.square().mean(1).sqrt()
        for m in torch.argsort(rms)[:4].tolist():
            grid22.append({
                "age": age_name, "value_column": value_column,
                "flag_columns": tuple(j for j in range(8) if (m >> j) & 1),
                "rms": float(rms[m]), "bias": float(errors[m].mean()),
                "nll": float(nlls[m])
            })
grid22 = pd.DataFrame(grid22).sort_values("rms")
print("=== EXPANDED REGISTER SEARCH ===")
print("Reference probe NLL:", probe21.iloc[:n22].nll.mean())
print(grid22.head(12).to_string(index=False))

print("\n=== ORIGINAL READOUTS ON EXPOSED HOURS ===")
dose_original22 = pd.read_csv("/work/fwd_2/dose_edit_original.csv")
exposed22 = dose_original22[dose_original22.reached.astype(bool)]
perstay22 = exposed22.groupby(["edit", "stay_id"]).shift.mean()
print(perstay22.groupby("edit").agg(["mean", "std", "count"]).to_string())
print("Per-hour shifts:")
print(exposed22.groupby(["edit", "hour"]).shift.mean().unstack(0).round(3).to_string())

print("\n=== RECORD-BASED OBJECT DEFINITIONS ===")
print("panel columns:", list(panel.columns))
print("panel index:", panel.index.names)
print(panel.head(4).to_string())
print("HC columns:", list(HC.columns))
print("Impulse coefficients:", irf_coefficients)
print("Dose residual summaries:")
for drug, residual_array in dose_residuals.items():
    r = np.asarray(residual_array)
    print(drug, "finite:", np.isfinite(r).sum(),
          "percentiles:", np.nanpercentile(r, [1, 10, 50, 90, 99]))

print("\n=== FORWARD SUMMARIES ===")
for k in [1, 2, 3, 4]:
    summary = json.loads(Path(f"/work/fwd_{k}/summary.json").read_text())
    print(k, summary)

with open("/work/notes.md", "a") as f:
    f.write("\n## Expanded register search — Python 22; forward 4\n"
            "- Binary flag changes alone did not resolve reconstruction.\n"
            "- Tested all eight value-column positions and all binary flag assignments.\n")
    f.write(grid22.head(6).to_string(index=False) + "\n")
    f.write("Original exposed-hour readouts:\n" +
            perstay22.groupby("edit").agg(["mean", "std", "count"]).to_string() + "\n")
```