The audit stopped because the vocabulary stores numeric specifications as a list, not a dictionary. No weights or features were changed. I’ll correct that schema assumption and repeat the check; this will tell us whether the apparent saturation comes from the model or from the local reconstruction.

```python
import dis, json, numpy as np, pandas as pd, torch

print("=== NUMERIC ENCODING ===")
numeric_text112 = json.dumps(VOCAB["numeric"])
print("codes:", VOCAB["codes"])
print("numeric type:", type(VOCAB["numeric"]).__name__)
print("numeric specifications:", numeric_text112[:1700])
if len(numeric_text112) > 1700:
    print("numeric specifications tail:", numeric_text112[-1400:])
print("numeric note:", str(VOCAB.get("numeric_note"))[:500])
print("repair_atom_base:", " | ".join(
    f"{i.opname}:{i.argrepr}" for i in dis.get_instructions(repair_atom_base)
)[:2900])

print("\n=== POLICY TARGET ALIGNMENT ===")
panel_pos111 = panel.index.get_indexer(pd.MultiIndex.from_arrays([
    latent_sid[drug_idx110],
    np.floor(latent_hour[drug_idx110] + 1e-5).astype(int)
], names=panel.index.names))
print("unmatched drug bundles:", int((panel_pos111 < 0).sum()))
assert (panel_pos111 >= 0).all()
innovation_targets111 = np.zeros((len(drug_idx110), 2))
policy_predictions111 = np.zeros_like(innovation_targets111)
for j111, drug111 in enumerate(["insulin", "norepinephrine"]):
    nf111 = dose_models[drug111][0].n_features_in_
    source111 = HC if HC.shape[1] == nf111 else H
    assert source111.shape[1] == nf111
    pred111 = np.zeros(len(drug_idx110))
    for fold111 in [0, 1]:
        take111 = fold[panel_pos111] == fold111
        pred111[take111] = dose_models[drug111][fold111].predict(
            source111.iloc[panel_pos111[take111]])
    act111 = drug_rates110[:, j111] > 0
    policy_predictions111[:, j111] = pred111
    innovation_targets111[act111, j111] = drug_rates110[act111, j111] - pred111[act111]
    print(drug111, "policy features", nf111, "active innovation mean/SD:",
          np.mean(innovation_targets111[act111, j111]),
          np.std(innovation_targets111[act111, j111]))

print("\n=== FULL-BUNDLE FEATURE VALIDATION ===")
variants111 = []
for label111, cc111, ee111, uu111 in [
    ("84", C84, E84, U84), ("85", c85, e85, u85)
]:
    hh111, zz111 = posterior109(cc111, ee111, uu111)
    use111 = np.asarray(valid84, dtype=bool)
    rms111 = float(torch.sqrt(((zz111[use111] - Ztarget84[use111])**2).mean()))
    print(label111, "posterior RMS", rms111)
    variants111.append((rms111, label111, hh111.numpy()))
variants111.sort(key=lambda q: q[0])
full_hidden111 = variants111[0][2].copy()
full_hidden111[drug_idx110] = drug_hidden110
full_targets111 = np.zeros((len(latent_sid), 2))
full_targets111[drug_idx110] = innovation_targets111
train_feature111 = (latent_sid % 5 != 4)
test_feature111 = ~train_feature111
print("Feature-fit train/test bundles:",
      int(train_feature111.sum()), int(test_feature111.sum()))

print("\n=== INSULIN x2 VERSUS x3 ===")
ins_active111 = drug_rates110[:, 0] > 0
diff23_111 = (hidden_edits110[("insulin", 3.0)] -
             hidden_edits110[("insulin", 2.0)])
print("mean incremental hidden norm:",
      np.linalg.norm(diff23_111[ins_active111], axis=1).mean())
print("relative to x2:",
      np.linalg.norm(diff23_111[ins_active111], axis=1).mean() /
      np.linalg.norm((hidden_edits110[("insulin", 2.0)] -
                      drug_hidden110)[ins_active111], axis=1).mean())
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 112 — corrected schema audit\n"
            "- Python 111 stopped on a list/dict assumption before creating features or editing weights.\n"
            "- Repeated numeric audit and aligned cross-fitted factual dose innovations.\n"
            "- Constructed full-bundle factual features and by-stay holdout. No weights changed.\n")
```