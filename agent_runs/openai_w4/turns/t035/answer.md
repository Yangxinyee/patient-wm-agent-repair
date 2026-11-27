The minimal probe confirms that the original rollouts barely respond despite record-based immediate effects. The insulin lag estimates also differ from the earlier single-decay fit, so I won’t use that fit as truth. Next I’ll inspect the lag moments and action embedding details; a stable lag pattern will guide the repair’s memory.

```python
import numpy as np
import pandas as pd

print("=== LAG MOMENTS ===")
for key, table in impulse_tables_repair.items():
    print(key)
    if isinstance(table, pd.DataFrame):
        print(table.head(8).to_string(index=False))
    else:
        print(type(table).__name__)

print("\n=== NUISANCE MODEL STRUCTURE ===")
for drug, models in dml_models.items():
    print(drug, type(models).__name__)
    if isinstance(models, dict):
        print({k: type(v).__name__ for k, v in models.items()})
    elif isinstance(models, (tuple, list)):
        print([type(v).__name__ for v in models])
print("history columns:", list(HX.columns))
print("panel index:", pan.index.names, "shape:", pan.shape)
print("dose quantiles:")
print(pan[["insulin", "norepinephrine"]].quantile([0, .5, .9, .99, 1]).to_string())

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for key in VOCAB:
    if any(word in key.lower() for word in ["numeric", "normal", "config", "unit"]):
        value = VOCAB[key]
        print(key, str(value)[:1200])

print("\n=== RELEVANT TENSOR NAMES ===")
for name in NAMES:
    if ("z_" in name or name.startswith(("embed.", "core.prelude", "core.flow.",
                                         "attr_heads.value_bin."))):
        print(name, SHAPES[name])

print("\n=== STATE SAMPLE ===")
for key in ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]:
    print(key, state_o[key][:8])
print("training stay range:", np.min(state_train_repair["stay_id"]),
      np.max(state_train_repair["stay_id"]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 29\n"
            "Recovered baseline shifts: insulin x2 -0.382, half +0.209; "
            "NE x2 +0.103, half -0.307. Immediate history-adjusted estimates "
            "are insulin -2.049 per rate and NE +3.171 per rate. "
            "Six-lag deconvolution suggests an insulin hump followed by decay "
            "rather than the earlier single-exponential fit; long-horizon "
            "truth remains provisional. No edits.\n")
```