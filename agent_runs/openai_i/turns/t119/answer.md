Routing END markers through the observation encoder reduced the reconstruction error about twentyfold, including on separate bundles. That supports this routing, though it is not exact. I’ll now measure hidden-feature changes under several dose multipliers and recover the factual innovation targets. Consistent dose sensitivity across multipliers would support a repair that also generalizes beyond doubling.

```python
import numpy as np, torch, pandas as pd, time

drug_cases110 = latent_bundles
drug_idx110 = np.array([c[0] for c in drug_cases110], dtype=int)
drug_ctx110 = latent_context(drug_idx110)
drug_e110 = torch.stack([
    pool109([end_atom99(r, best109[2]) for r in c[2]], "encode")
    for c in drug_cases110
])
drug_u110 = torch.stack([
    pool109([latent_start_atom(r) for r in c[1]], "encode_action", best109[3])
    for c in drug_cases110
])
drug_hidden110, drug_z110 = posterior109(drug_ctx110, drug_e110, drug_u110)
drug_rates110 = np.array([
    [sum(float(r["rate"]) for r in c[1] if r["event"] == drug)
     for drug in ["insulin", "norepinephrine"]]
    for c in drug_cases110
])
hidden_edits110 = {}
t110 = time.time()
for drug110 in ["insulin", "norepinephrine"]:
    for factor110 in [0.0, 0.5, 1.5, 2.0, 3.0]:
        vectors110 = []
        for case110 in drug_cases110:
            atoms110 = []
            for row110 in case110[1]:
                rr110 = dict(row110)
                if rr110["event"] == drug110:
                    rr110["rate"] = float(rr110["rate"]) * factor110
                    rr110["dose"] = float(rr110["dose"]) * factor110
                atoms110.append(latent_start_atom(rr110))
            vectors110.append(pool109(atoms110, "encode_action", best109[3]))
        hh110, _ = posterior109(drug_ctx110, drug_e110, torch.stack(vectors110))
        hidden_edits110[(drug110, factor110)] = hh110.numpy()
drug_hidden110 = drug_hidden110.numpy()
print("Factual drug-bundle posterior RMS:",
      float(torch.sqrt(((drug_z110 -
          torch.from_numpy(latent_state[drug_idx110, -32:]))**2).mean())))
print("Dose-feature construction seconds:", round(time.time()-t110, 1))
for j110, drug110 in enumerate(["insulin", "norepinephrine"]):
    active110 = drug_rates110[:, j110] > 0
    print(drug110, "active bundles", int(active110.sum()))
    for factor110 in [0.0, 0.5, 1.5, 2.0, 3.0]:
        dh110 = hidden_edits110[(drug110, factor110)] - drug_hidden110
        print(" factor", factor110, "mean hidden-change norm",
              round(float(np.linalg.norm(dh110[active110], axis=1).mean()), 5))

print("\n=== INNOVATION OBJECTS ===")
for drug110, obj110 in dose_models.items():
    print(drug110, type(obj110).__name__)
    if isinstance(obj110, dict):
        for key110, val110 in obj110.items():
            print(" ", key110, type(val110).__name__,
                  tuple(val110.shape) if hasattr(val110, "shape")
                  else str(val110)[:100])

print("\n=== CACHED FULL-BUNDLE FEATURES ===")
for name110, obj110 in list(globals().items()):
    if (hasattr(obj110, "shape") and
        (name110.endswith("84") or name110.endswith("85") or
         "innovation" in name110.lower())):
        print(name110, type(obj110).__name__, tuple(obj110.shape))
print("State stays in train/tuning:",
      np.isin(latent_stays, train.stay_id.unique()).sum(),
      np.isin(latent_stays, tuning.stay_id.unique()).sum())
print("Panel index:", panel.index.names, "columns:", list(panel.columns))
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 110 — dose-feature construction\n"
            "- END-as-observation routing passed separate-bundle validation at RMS about 0.0043.\n"
            "- Constructed posterior hidden features for factual doses and factors 0, .5, 1.5, 2, 3.\n"
            "- These perturbations calibrate dose encoding, not drug-effect truth.\n"
            "- Inspecting retained factual innovation targets. No weights changed.\n")
```