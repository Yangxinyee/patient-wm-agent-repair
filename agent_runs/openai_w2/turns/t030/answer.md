The full-sample reconstruction also matches the original predictions without meaningful bias. The probe failed only because weights are PyTorch tensors, not NumPy arrays; no weights were changed. I’ll fix that copy operation, then measure the probe’s state response.

```python
import torch
from pathlib import Path

assert edited() == []
probe_numeric_original = load_original("embed.numeric.weight")
probe_numeric = probe_numeric_original.clone()
probe_delta_z = 0.5
probe_numeric[:, 3] += probe_delta_z * probe_numeric_original[:, 2]
probe_numeric[:, 5] += probe_delta_z * probe_numeric_original[:, 4]
save_edit("embed.numeric.weight", probe_numeric)

probe_rate_changes = {
    drug: float(probe_delta_z * numeric_specs[(drug, "rate")]["iqr"])
    for drug in ["insulin", "norepinephrine"]
}
print("Rate-equivalent increments:", probe_rate_changes)
print("Edited tensors:", edited())
print("Stays:", sample_stays)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 25–26: successful head reconstruction and diagnostic dose probe\n"
        "All 4660 reading means reconstructed without meaningful channel bias "
        "(glucose_finger -0.040, lab -0.091, MAP arterial -0.0002, cuff +0.003).\n"
        "Head config: GELU for gap/prior/value; query type 1; status 0; "
        "last flags (0,), age column 3, value-z column 6; age log1p(seconds)/10; "
        "bin side left. Original states: S_reference; baseline head features: "
        "head_original_features, 64 prior draws, seed 917.\n"
        "Run 25 stopped before saving because load_original returns torch.Tensor; "
        "run 26 fixed with clone(). Temporary edit ONLY embed.numeric.weight: "
        "presence columns 3/5 receive 0.5 times numeric dose/rate columns 2/4. "
        "Equivalent rate increments: " + repr(probe_rate_changes) + ".\n"
        "Restore embed.numeric.weight after obtaining probe states. Bins are "
        "unchanged; the final correction must be tested with exact dose_edit.\n"
        "F includes current drug exposures: exclude the target drug's current "
        "exposure when estimating its assignment from pre-dose history.\n"
    )
```