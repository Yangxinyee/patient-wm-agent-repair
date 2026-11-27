The records contain dose variation beyond the apparent threshold policy, and two analyses agree on an immediate insulin effect near −2 per rate unit. That is encouraging, but the sustained response remains uncertain. I’ll inspect the lagged estimates and observation patterns; agreement across horizons and methods would justify a rollout target.

```python
import numpy as np
import pandas as pd

print("=== HOURLY PANEL ===")
print("index:", pan.index.names, "columns:", list(pan.columns))
print(pan.head(8).to_string())
print("history columns:", list(HX.columns))

print("\n=== INNOVATION ESTIMATES ===")
print(dml_results.to_string(index=False))
for drug, obj in dml_residuals.items():
    print(drug, type(obj).__name__)
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(" ", key, type(value).__name__, getattr(value, "shape", None))
    else:
        print("shape:", getattr(obj, "shape", None))

print("\n=== LAGGED MOMENTS ===")
for drug, obj in impulse_tables_repair.items():
    print(drug)
    print(obj.to_string(index=False) if isinstance(obj, pd.DataFrame) else obj)

print("\n=== SELECTED RETAINED CAUSAL OBJECTS ===")
for name in sorted(list(globals())):
    if any(term in name.lower() for term in
           ["kernel", "persist", "deconv", "moment", "impulse", "rho", "causal"]):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, str(obj)[:1100])
        elif isinstance(obj, (np.ndarray, pd.DataFrame)):
            print(name, getattr(obj, "shape", None),
                  np.asarray(obj).reshape(-1)[:18])
        elif np.isscalar(obj):
            print(name, obj)

print("\n=== NOREPINEPHRINE POLICY ===")
print(map_policy_table.head(12).to_string(index=False))

print("\n=== MODEL RECONSTRUCTION RECIPES ===")
print("head:", head_cfg_repair)
print("drug atoms:", atom_recipe_repair)
print("measurement atoms:", globals().get("r45_measure_recipes",
                                       globals().get("measurement_recipes_repair", "not found")))
print("available reconstruction variables:",
      ", ".join(k for k in sorted(globals())
                if k.startswith(("r44_", "r45_", "rp_"))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 47–48\n"
            "Recovered threshold and innovation evidence for immediate effects, but sustained "
            "response is not yet accepted as ground truth. Inspecting lagged moments and "
            "observation selection before choosing a target. No edits.\n")
```
