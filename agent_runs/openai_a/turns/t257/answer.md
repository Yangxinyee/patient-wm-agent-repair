The second batch’s NLL increase is 0.00529; together, the two batches average almost exactly 0.005. Scale 7 therefore has insufficient safety margin. I’ll restore scale 6 and inspect whether a record-based calibration can reduce factual cost without weakening the insulin response. Independent factual improvement with retained rollout effects would support that approach.

```python
from pathlib import Path

for _repair_name, _repair_tensor in stronger6_checkpoint.items():
    save_edit(_repair_name, _repair_tensor)

with open("/work/notes.md", "a") as f:
    f.write("""
## Scale-7 safety rejection; Python 222, forwards 26/50
- Second-100 tuning factual: original 1.62044, edited 1.62573, delta +0.00529.
- Reading-weighted first-200 delta approximately +0.00499: inadequate safety margin.
- Restored stronger6_checkpoint. scale7_checkpoint retained only as an unaccepted trial.
- Investigating empirical calibration or a more efficient contrast before further scaling.
""")

print("RESTORED EDITS:", edited())

for _repair_fn_name in (
    "make_contrast_graft", "make_selective_contrast_graft",
    "selective_cache", "cache_losses", "screen_value_candidate"
):
    _repair_fn = globals()[_repair_fn_name]
    print("\nFUNCTION", _repair_fn_name)
    print("Referenced globals:", _repair_fn.__code__.co_names)
    print("Local names:", _repair_fn.__code__.co_varnames)

print("\nCANDIDATE/CACHE VARIABLE NAMES")
print(sorted(n for n in list(globals()) if not n.startswith("_") and
             (n.startswith(("r3", "sc_", "selective_")) or
              any(s in n for s in ("cache", "contrast_weights", "whiten")))))

for _repair_var_name in (
    "cache_registry_recovery", "trial_contrast_weights",
    "selective_trial", "rd_dynamic_fit", "base_rd"
):
    if _repair_var_name not in globals():
        continue
    _repair_var = globals()[_repair_var_name]
    print("\nVARIABLE", _repair_var_name, type(_repair_var).__name__)
    if isinstance(_repair_var, dict):
        for _repair_k, _repair_v in _repair_var.items():
            print(_repair_k, type(_repair_v).__name__,
                  getattr(_repair_v, "shape", None),
                  list(_repair_v.keys())[:12] if isinstance(_repair_v, dict) else "")
    elif hasattr(_repair_var, "shape"):
        print("shape:", _repair_var.shape)
    else:
        print(str(_repair_var)[:700])

print("\nNOTEBOOK IDENTIFICATION CONTINUED")
print(Path("/work/notes.md").read_text()[6800:12000])
```
