The insulin estimate is supported by dose-policy discontinuities near 180 and 220, where reading availability does not jump. The norepinephrine estimate is less secure. I’ll now calculate how a small drug-specific latent change affects the next reading, using the retained reconstruction only for sensitivity estimates. Agreement across small perturbations would support a reversible posterior edit; unstable or cross-outcome effects would argue against it.

```python
import inspect
import numpy as np
import torch

print("=== HELPER IMPLEMENTATIONS ===")
for fn44 in [core_update_local, core_attn, core_layer, head_calc,
             decoder_query, recon_record_atom, _post33]:
    print("\n", fn44.__name__, inspect.signature(fn44))
    try:
        src44 = inspect.getsource(fn44)
        print(src44[:1700])
    except Exception:
        print("names:", fn44.__code__.co_names)
        print("locals:", fn44.__code__.co_varnames)
        print("constants:", [v for v in fn44.__code__.co_consts
                              if isinstance(v, (str, int, float, tuple))][:22])
print("=== CORE SAMPLE ===")
print(state_df.iloc[np.asarray(core_rows)[:12]].to_string(index=False))
print("h RMS:", float(np.sqrt(np.mean(core_target_h ** 2))))
print("out RMS:", float(np.sqrt(np.mean(core_target_out ** 2))))
print("=== HEAD META ===")
print(head_meta.head(3).to_string(index=False))
print("=== POSTERIOR CONTRAST FIT ===")
print("cf_scale min/median/max:", np.quantile(cf_scale, [0, .5, 1]))
for key44, value44 in cf_fits.items():
    print(key44, "coefficient norms", np.linalg.norm(value44[:-1], axis=0),
          "intercept", value44[-1])
print("=== STATE/MEMORY TYPES ===")
for key44 in ["S0", "core_memories", "cf_starts", "cf_ends"]:
    obj44 = globals()[key44]
    print(key44, [(str(k), type(v).__name__, getattr(v, "shape", None))
                  for k, v in list(obj44.items())[:5]])
print("=== PRIOR TENSORS ===")
print([(name, SHAPES[name]) for name in NAMES
       if "prior" in name or "gap" in name or "type_emb" in name])
print("=== PROVISIONAL SCHEDULE READOUTS ===")
for drug44, outcome44, beta44, phi44 in [
    ("insulin", "glucose_finger", float(dyn_fit.x[0]), float(dyn_fit.x[1])),
    ("norepinephrine", "map_arterial", 3.248868, .85)
]:
    try:
        print(drug44, schedule_readout(beta44, phi44, drug44, outcome44))
    except Exception as e44:
        print(type(e44).__name__, str(e44)[:160])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 44: downstream sensitivity preparation\n"
            "Insulin identification is supported by non-selection thresholds 180/220; NE remains provisional. "
            "Inspecting local core and head helpers to choose a reversible drug-posterior direction. "
            "No edits. Python runs: 44; forward passes: 3.\n")
```
