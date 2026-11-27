The validated transition exposes the posterior inputs we need. I’ll test whether a small regression on those inputs can recover history-adjusted dose innovations and respond correctly to halved or doubled doses. Held-out stays will be decisive: fitting the training innovations alone would not justify an edit.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.linalg import solve

idx82 = np.asarray(valid73, dtype=int)
ctx82, hf82, dt82 = pre_context_repair(repair_bank, idx82)
zp_input82 = torch.cat([ctx82, E65[idx82], U65[idx82]], dim=-1)
zp_hidden82 = F.gelu(rp_linear(zp_input82, "z_post.0")).numpy().astype(np.float64)
zp_mu82 = zp_hidden82.mean(axis=0)
Z82 = np.c_[zp_hidden82 - zp_mu82, np.ones(len(idx82))]
bundle_to_row82 = np.full(len(repair_bank["x"]), -1, dtype=int)
bundle_to_row82[idx82] = np.arange(len(idx82))
labels82 = {}
weights82 = {}
eps_lookup82 = {}
for drug in ["insulin", "norepinephrine"]:
    obj = dml_residuals[drug]
    ii = np.asarray(obj["indices"])
    ii = np.flatnonzero(ii) if ii.dtype == bool else ii.astype(int).reshape(-1)
    eps_lookup82[drug] = pd.Series(np.asarray(obj["rd"]).reshape(-1),
                                   index=pan.index.take(ii))
    rr = repair_records.loc[repair_records.event.eq(drug)]
    keys = pd.MultiIndex.from_arrays(
        [rr.stay_id.to_numpy(), np.floor(rr.hour.to_numpy()).astype(int)],
        names=pan.index.names)
    ee = eps_lookup82[drug].reindex(keys).to_numpy()
    br = bundle_to_row82[rr.bundle_index.to_numpy(dtype=int)]
    target = np.zeros(len(idx82))
    weight = np.ones(len(idx82))
    present = br >= 0
    weight[br[present]] = 0.0
    good = present & np.isfinite(ee)
    target[br[good]] = ee[good]
    weight[br[good]] = 1.0
    labels82[drug] = target
    weights82[drug] = weight
    print(drug, "labelled actions", good.sum(),
          "innovation SD", np.std(ee[good]),
          "missing action labels", (present & ~np.isfinite(ee)).sum())

cfZ82, cfdr82 = {}, {}
action_mask82 = repair_records.event.isin(["insulin", "norepinephrine"]).to_numpy()
for drug in ["insulin", "norepinephrine"]:
    drug_mask = repair_records.event.eq(drug).to_numpy()
    for factor in [0.5, 2.0]:
        atoms_cf = atoms65.clone()
        rr = repair_records.loc[drug_mask]
        atoms_cf[drug_mask] = atom_exact65(
            drug, rr.value.to_numpy(), rr.rate.to_numpy() * factor,
            rr.dose.to_numpy() * factor)
        uu = pooled_encoder58(
            atoms_cf[action_mask82],
            repair_records.loc[action_mask82, "bundle_index"].to_numpy(),
            counts_u58, "encode_action")
        hidden = F.gelu(rp_linear(torch.cat(
            [ctx82, E65[idx82], uu[idx82]], dim=-1), "z_post.0")).numpy()
        cfZ82[(drug, factor)] = np.c_[hidden - zp_mu82, np.ones(len(idx82))]
        rate = np.zeros(len(repair_bank["x"]))
        np.add.at(rate, rr.bundle_index.to_numpy(dtype=int), rr.rate.to_numpy())
        cfdr82[(drug, factor)] = rate[idx82] * (factor - 1.0)

sid82 = np.asarray(repair_bank["stay_id"])[idx82]
fit82 = sid82 < np.unique(sid82)[60]
test82 = ~fit82
decoder_candidates82 = {}
rows82 = []
for drug in labels82:
    yy, ww = labels82[drug], weights82[drug]
    aa = np.vstack([cfZ82[(drug, factor)] - Z82 for factor in [0.5, 2.0]])
    bb = np.concatenate([cfdr82[(drug, factor)] for factor in [0.5, 2.0]])
    active = np.abs(bb) > 1e-8
    afit = active & np.tile(fit82, 2)
    X = Z82[fit82]
    w = ww[fit82]
    H = (X.T * w) @ X / w.sum()
    g = X.T @ (w * yy[fit82]) / w.sum()
    HA = aa[afit].T @ aa[afit] / afit.sum()
    ga = aa[afit].T @ bb[afit] / afit.sum()
    rs = np.trace(H) / H.shape[0]
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for strength in [0.0, 1.0, 10.0]:
            coef = solve(H + strength * HA +
                         alpha * rs * np.eye(H.shape[0]),
                         g + strength * ga, assume_a="pos")
            decoder_candidates82[(drug, alpha, strength)] = coef
            held = test82 & (ww > 0)
            action_held = held & (np.abs(yy) > 1e-8)
            pred = Z82 @ coef
            r2 = 1.0 - np.sum((pred[action_held] - yy[action_held])**2) / \
                np.sum((yy[action_held] - yy[action_held].mean())**2)
            row = {"drug": drug, "alpha": alpha, "strength": strength,
                   "held_action_R2": r2,
                   "held_all_RMSE": np.sqrt(np.mean((pred[held]-yy[held])**2)),
                   "nonaction_RMS": np.sqrt(np.mean(pred[held & (yy == 0)]**2))}
            for factor in [0.5, 2.0]:
                delta = (cfZ82[(drug, factor)] - Z82) @ coef
                dr = cfdr82[(drug, factor)]
                mm = test82 & (np.abs(dr) > 1e-8)
                row[f"cf_slope_{factor}"] = np.dot(delta[mm], dr[mm]) / np.dot(dr[mm], dr[mm])
            rows82.append(row)
decoder_table82 = pd.DataFrame(rows82)
print(decoder_table82.round(4).to_string(index=False))

print("\nEND recipe sample:")
print(all_ends65[["event", "value", "rate", "dose", "end_hour"]].head(6).to_string(index=False))
print("End-atom absent-attribute RMSE:")
for event in ["insulin", "norepinephrine", "hospital_admission", "icu_admission"]:
    mm = all_ends65.event.eq(event).to_numpy()
    if mm.any():
        n = mm.sum()
        aa = atom_exact65(event, np.full(n, np.nan), np.full(n, np.nan),
                          np.full(n, np.nan), kind=2)
        print(event, float(torch.sqrt(torch.mean((aa-end_atoms65[mm])**2))))
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 82\n"
            "Tested posterior-hidden dose-innovation regressions with "
            "60/40 stay-separated validation and controlled half/double "
            "response constraints. No weight edits; decoder transfer and "
            "non-action leakage determine whether a posterior repair is plausible.\n")
```
