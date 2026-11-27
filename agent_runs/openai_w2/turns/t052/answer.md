The recovered record checks support an insulin decrease of roughly 1.55–1.97 per additional rate unit, so its near-zero response to amplification needs fixing. Norepinephrine did respond to the numeric edit; extrapolating that response suggests a scale near 12. I’ll test that candidate while preserving each drug’s average recorded embedding. Factual degradation or response saturation would argue against keeping it.

```python
import numpy as np
import torch
from pathlib import Path

def install_numeric_amplification(scale):
    nw = load_original("embed.numeric.weight").clone()
    cw = load_original("embed.code.weight").clone()
    original_nw = nw.clone()
    nw[:, 2] *= float(scale)
    nw[:, 4] *= float(scale)
    centers = {}
    for drug_name in ["insulin", "norepinephrine"]:
        events = train.loc[train.event.eq(drug_name)]
        center_vector = torch.zeros(192, dtype=torch.float32)
        centers[drug_name] = {}
        for attribute, column in [("dose", 2), ("rate", 4)]:
            spec = numeric_specs[(drug_name, attribute)]
            z = (events[attribute].to_numpy(dtype=float) - spec["median"]) / spec["iqr"]
            mean_z = float(np.mean(z[np.isfinite(z)]))
            centers[drug_name][attribute] = mean_z
            center_vector += original_nw[:, column] * mean_z
        cw[code_ids[drug_name]] -= (float(scale) - 1.0) * center_vector
    save_edit("embed.numeric.weight", nw)
    save_edit("embed.code.weight", cw)
    return centers

print("=== ORIGINAL BIN EMBEDDING NORMS ===")
for name in ["embed.bins.dose.weight", "embed.bins.rate.weight"]:
    weight = np.asarray(load_original(name))
    print(name, np.linalg.norm(weight, axis=1).round(3).tolist())
print("=== LATENT FIT DIAGNOSTICS ===")
for drug_name, fit in latent_fits.items():
    print(drug_name, "stays", len(fit.get("stay_ids", [])),
          "parameters", fit.get("parameters"))
print("=== INSTALLED NUMERIC SCALE 12 ===")
numeric_scale12_centers = install_numeric_amplification(12.0)
print(numeric_scale12_centers)
print("Current edits:", edited())
for name in edited():
    delta = np.asarray(load(name)) - np.asarray(load_original(name))
    print(name, "delta_norm", float(np.linalg.norm(delta)))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 45: numeric scale-12 diagnostic\n"
            "Recovered clean insulin lab RD about -1.55/unit and rich history estimate -1.97/unit. "
            "Scale-3 rollout changed NE substantially but not insulin. "
            "Installed numeric scale 12, chosen from observed NE response and provisional record target; "
            "per-drug code offsets preserve the mean training atom embedding. "
            "This remains a reversible diagnostic, awaiting factual and paired rollout tests.\n")
```
