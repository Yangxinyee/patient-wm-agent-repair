The recovered diagnostics rejected a head-only correction because it could not recover enough dose response without distorting factual predictions. I’ll focus on the action input instead. Before choosing the next amplification, I’ll check the dose normalisation, bin embeddings, and causal-estimate uncertainty. This should distinguish a weak numeric channel from a misleading target estimate.

```python
from pathlib import Path
import numpy as np

text = Path("/work/notes.md").read_text()
for heading in ["## Run 5:", "## Run 6:", "## Run 10:", "## Run 12:",
                "## Runs 25", "## Run 27:", "## Run 32:"]:
    pos = text.find(heading)
    if pos >= 0:
        end = text.find("\n## ", pos + 4)
        print(text[pos:end if end >= 0 else len(text)][:750])

print("=== ACTION ENCODER TENSORS ===")
for name in NAMES:
    if name.startswith("encode_action.") or name in {
        "embed.bins.dose.weight", "embed.bins.rate.weight",
        "embed.numeric.weight", "embed.numeric.bias"
    }:
        print(name, SHAPES[name])

print("=== NORMALISATION AND EXPOSURE ===")
for drug in ["insulin", "norepinephrine"]:
    events = train.loc[train.event.eq(drug)]
    for attribute in ["dose", "rate"]:
        spec = numeric_specs[(drug, attribute)]
        zz = (events[attribute].to_numpy() - spec["median"]) / spec["iqr"]
        print(drug, attribute, "median", spec["median"], "iqr", spec["iqr"],
              "mean_z", float(np.mean(zz)), "z_quantiles",
              np.quantile(zz, [0, .1, .5, .9, 1]).round(3).tolist())
    center = np.asarray(action_centers[drug])
    print(drug, "numeric-center norm", float(np.linalg.norm(center)))

print("=== ORIGINAL EMBEDDING MAGNITUDES ===")
for name in ["embed.numeric.weight", "embed.bins.dose.weight", "embed.bins.rate.weight"]:
    weight = np.asarray(load_original(name))
    print(name, "row_norms", np.linalg.norm(weight, axis=1).round(3).tolist()
          if weight.shape[0] == 17 else
          "column_norms=" + str(np.linalg.norm(weight, axis=0).round(3).tolist()))
print("Previous constant-z probe:", probe_delta_z, probe_rate_changes)
print("=== LATENT FIT SIZES ===")
for drug, fit in latent_fits.items():
    print(drug, "keys", list(fit),
          "stays", len(fit.get("stay_ids", [])),
          "parameters", fit.get("parameters"))
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 40–44: restart recovery\n"
            "Architecture and retained helpers recovered. Prior head-only correction failed: "
            "drug response was poorly recoverable under factual-zero constraints. "
            "Current scale-3 edit acts only on numeric dose/rate columns, not fine-bin embeddings. "
            "Investigating nonlinear action encoding and bin pathways before stronger edits.\n")
```
