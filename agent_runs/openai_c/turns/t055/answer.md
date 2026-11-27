The saved MAP sensor is noisy on factual states: a nominal +0.5 response also produces substantial unrelated variation. That makes stronger versions risky. I’ll measure how much the reading head relies on the world state versus its last-reading register, to assess whether their relative gains can change without shifting factual predictions.

```python
print("WORLD-STATE / REGISTER SURROGATE DECOMPOSITION")
for event, fit in head_surrogates.items():
    coef = np.asarray(fit["ridge"].coef_).reshape(-1)
    O, R = fit["O"], fit["R"]
    world_part = O @ coef[:O.shape[1]]
    register_part = R @ coef[O.shape[1]:]
    frame = fit["frame"]
    valid = np.isfinite(frame["last_value"].to_numpy())
    last_v = frame["last_value"].to_numpy()[valid]
    design = np.column_stack([np.ones(valid.sum()), last_v])
    fitted_world = np.linalg.lstsq(design, world_part[valid], rcond=None)[0]
    print(event, {
        "world_sd": float(world_part.std()),
        "register_sd": float(register_part.std()),
        "world_register_corr": float(np.corrcoef(world_part, register_part)[0, 1]),
        "world_on_last_value": fitted_world.tolist(),
        "world_residual_sd": float(np.std(world_part[valid] - design @ fitted_world)),
        "register_coefficients": coef[O.shape[1]:].tolist(),
    })

print("\nLATENT AND QUERY TENSORS")
print({k: SHAPES[k] for k in NAMES
       if any(s in k for s in ["z_", "z.", "query_norm", "last_numeric", "type_emb"])})

print("\nDOSE-Z DIAGNOSTICS")
for drug in repair_signal:
    print(drug, repair_signal[drug].get("z"))

print("\nLAST-READING FEATURE RANGES")
for event, fit in head_surrogates.items():
    print(event, np.round(np.quantile(fit["R"], [0.1, 0.5, 0.9], axis=0), 3).tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 36\n"
            "- Direct NE head sensor has factual SD about 1.8 rate units, too noisy for a large extrapolation.\n"
            "- Inspecting world-state versus register contributions before considering a compensated query-gain edit.\n"
            "- Numeric gain 4 remains installed. Python runs: 36; forward passes: 19.\n")
```
