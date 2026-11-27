The local recurrent step differs substantially from the saved GPU states, so I won’t use its sensitivities yet. I’ll check its memory and time conventions. A consistently small reconstruction error would make local screening useful; otherwise, candidates must be judged directly with paired GPU tests.

```python
import itertools, time

print("=== RECOVERABLE STEP SOURCE ===")
for k, v in list(globals().items()):
    if isinstance(v, str) and "def local_step67(" in v and len(v) < 20000:
        a = v.find("def local_step67(")
        print(k, v[a:a+4200])
        break

j89 = probe_rows88[0]
z89 = torch.as_tensor(s4["x"][j89,960:992]).reshape(1,32)
hs89 = torch.as_tensor(s4["x"][j89,192:960]).reshape(1,4,192)
oldmode89, oldpack89 = wm_memory_mode, wm_kv_layout
trial89 = []
for mode, pack, incl, units in itertools.product(
        ["unnormalized", "norm_then_age", "age_then_norm"],
        ["global", "interleaved"], [False, True], [1.0, 3600.0]):
    try:
        wm_memory_mode, wm_kv_layout = mode, pack
        hh = local_step67(j89, z89, include_current=incl, age_mode="input",
                          age_units=units, age_scale=1.0)
        err = float(((hh-hs89)**2).mean().sqrt())
        trial89.append((err, mode, pack, incl, units))
    except Exception as e:
        trial89.append((float("inf"), mode, pack, incl, units))
wm_memory_mode, wm_kv_layout = oldmode89, oldpack89
print("\nBEST STEP CONVENTIONS", sorted(trial89)[:12])
print("Saved state RMS", float((hs89**2).mean().sqrt()))
print("Embedding/state reference shapes", wm_e4.shape, U4.shape)

print("\n=== ENCODER HIDDEN AND PANEL OBJECTS ===")
for nm in ["panel", "panel_tune", "train", "tuning", "df_train", "df_tune",
           "predI", "predNE", "nuisance_models", "treatment_models", "effect_models"]:
    if nm in globals():
        ob=globals()[nm]
        print(nm, type(ob).__name__, getattr(ob,"shape",None),
              list(ob.columns)[:18] if isinstance(ob,pd.DataFrame) else "")
print("Current edits", edited())
```
