All 57 edited tensors were stale in the replay dictionary. Reloading them restored the intended private-coordinate pairs and brought the first dose signal close to the GPU result. The cached world embeddings may also be stale. I’ll refresh and check them against freshly encoded bundles before testing finite-dose responses.

```python
import numpy as np, torch

wm_e4_before184 = wm_e4.clone()
gamma_old184 = tensor183(load_original("encode.norm.weight"))
beta_old184 = tensor183(load_original("encode.norm.bias"))
gamma_new184 = tensor183(load("encode.norm.weight"))
beta_new184 = tensor183(load("encode.norm.bias"))

candidate_e184 = ((wm_e4_before184-beta_old184)
                  / gamma_old184 * gamma_new184 + beta_new184)
changed184 = torch.nonzero(
    (gamma_old184 != gamma_new184) | (beta_old184 != beta_new184)
).reshape(-1)
print("=== WORLD ENCODER NORMALIZATION CHANGES ===")
print("changed coordinates", changed184.tolist())

print("\n=== CACHE VERSUS FRESH WORLD ENCODING ===")
fresh_checks184 = []
for j184 in [3,4,6,10,30,31]:
    sid184 = int(s4["stay_id"][j184])
    minute184 = int(round(float(s4["hour"][j184])*60))
    tab184 = groups4.get((sid184,minute184))
    if tab184 is None:
        continue
    aa184 = []
    unsupported184 = []
    for row184 in tab184.itertuples(index=False):
        if row184.event in ["insulin", "norepinephrine"]:
            continue
        if row184.event in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
            aa184.append(measurement_atom_local(row184.event,[row184.value]))
        else:
            unsupported184.append(row184.event)
    if unsupported184:
        print(j184, "visit bundle skipped", unsupported184)
        continue
    atoms184 = torch.cat(aa184,dim=0) if aa184 else torch.empty((0,192))
    fresh184 = wm_set(atoms184,"encode").reshape(192)
    err184 = float(torch.sqrt(torch.mean((candidate_e184[j184]-fresh184)**2)))
    fresh_checks184.append(err184)
    print(j184, "old cache RMSE",
          float(torch.sqrt(torch.mean((wm_e4_before184[j184]-fresh184)**2))),
          "refreshed cache RMSE", err184)

if fresh_checks184 and max(fresh_checks184) < 0.001:
    wm_e4 = candidate_e184
    print("Accepted refreshed world embedding cache.")
else:
    print("Cache transformation not validated; retained existing cache.")

print("\n=== CURRENT-CHECKPOINT BOUNDARY CHECK ===")
trials184 = []
for memory184 in ["unnormalized","norm_then_age",
                  "age_then_norm","age_then_plainnorm"]:
    wm_memory_mode = memory184
    for include184 in [False,True]:
        errors184 = []
        for j184 in [3,4,30,31]:
            pred184 = wm_step_boundary176(
                j184,include_current=include184,
                age_mode="input",age_units="seconds"
            ).detach().cpu().numpy().reshape(4,192)
            truth184 = s4["x"][j184,192:960].reshape(4,192)
            errors184.append(float(np.sqrt(np.mean(
                (pred184[:,ordinary182]-truth184[:,ordinary182])**2))))
        trials184.append((float(np.mean(errors184)),memory184,include184,errors184))
for row184 in sorted(trials184)[:4]:
    print(row184)
best184 = min(trials184)
wm_memory_mode = best184[1]
print("best", best184)

with open("/work/notes.md","a") as f184:
    f184.write("\n## Runs 183–184: full checkpoint audit\n"
               "- W0 contained stale versions of all 57 edited tensors; reloaded every tensor from current edits.\n"
               "- Private-pair algebra now agrees with the GPU, with close first-dose amplitude.\n"
               "- Audited cached world embeddings against fresh current-weight encodings.\n"
               "- No model weights changed; earlier CPU finite-dose diagnostics require rechecking.\n")
```