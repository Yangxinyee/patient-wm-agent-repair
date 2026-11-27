The mismatch has a concrete cause: `s4` contains original-model states, while the replay uses edited weights. That comparison cannot validate the transport channel. I’ll switch to edited-model states and repeat the boundary checks. A much smaller error would confirm checkpoint mismatch as the main problem.

```python
import numpy as np, torch

s4_original182 = s4
with np.load("/work/fwd_15/states_edited.npz") as f182:
    s4 = {k: f182[k].copy() for k in f182.files}

idx182 = np.array([61, 21, 15, 66])
ordinary182 = np.setdiff1d(np.arange(192), idx182)
print("=== EDITED STATE PRIVATE COORDINATES ===")
for j182 in [2, 3, 4]:
    hh182 = s4["x"][j182,192:960].reshape(4,192)
    print(j182, "hour", float(s4["hour"][j182]),
          "mean h", hh182.mean(0)[idx182].tolist(),
          "out", s4["x"][j182,:192][idx182].tolist())
print("world vector at action-only bundle",
      "RMS", float(torch.sqrt(torch.mean(wm_e4[3].square()))),
      "difference from empty_world176",
      float(torch.sqrt(torch.mean((wm_e4[3]-empty_world176).square()))))

print("\n=== ALIGNED BOUNDARY REPLAY ===")
old_settings182 = (wm_memory_mode, wm_kv_layout)
wm_kv_layout = "global"
trials182 = []
js182 = [3, 4, 30, 31]
for memory182 in ["unnormalized", "norm_then_age",
                  "age_then_norm", "age_then_plainnorm"]:
    wm_memory_mode = memory182
    for include182 in [False, True]:
        errs182, private_errs182 = [], []
        for j182 in js182:
            pred182 = wm_step_boundary176(
                j182, include_current=include182,
                age_mode="input", age_units="seconds"
            ).detach().cpu().numpy().reshape(4,192)
            true182 = s4["x"][j182,192:960].reshape(4,192)
            errs182.append(float(np.sqrt(np.mean(
                (pred182[:,ordinary182]-true182[:,ordinary182])**2))))
            private_errs182.append(float(np.sqrt(np.mean(
                (pred182[:,idx182]-true182[:,idx182])**2))))
        trials182.append((float(np.mean(errs182)), memory182, include182,
                          errs182, private_errs182))
for result182 in sorted(trials182)[:5]:
    print(result182)
best182 = min(trials182)
wm_memory_mode = best182[1]
print("best settings", wm_memory_mode, wm_kv_layout,
      "include current", best182[2])

print("\n=== AVAILABLE ATOM/ENCODER HELPERS ===")
for nm, obj in sorted(list(globals().items())):
    if (callable(obj) and hasattr(obj, "__code__")
        and any(s in nm.lower() for s in ["atom", "world", "encode"])
        and not nm.startswith("_")):
        print(nm, obj.__code__.co_varnames[:obj.__code__.co_argcount],
              obj.__code__.co_names[:12])

with open("/work/notes.md", "a") as f182:
    f182.write("\n## Run 182: checkpoint alignment\n"
               "- Found s4 matched ORIGINAL states in fwd_15, whereas replay weights were EDITED.\n"
               "- Earlier CPU/GPU discrepancies therefore do not establish a recurrence or transport defect.\n"
               "- Switched s4 to fwd_15 edited states and repeated boundary/memory checks; no weights changed.\n")
```