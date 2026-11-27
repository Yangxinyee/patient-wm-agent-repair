The dose residuals and latent directions are available, and drug-bundle posterior reconstruction has RMS error about 0.0042. I’ll fit a small linear innovation decoder using recorded residuals and dose-encoding perturbations, with a stay-level validation split. Near-unit responses to changed rates and near-zero responses on non-drug bundles would support testing a reversible weight edit.

```python
from sklearn.linear_model import Ridge

# Build factual features for every exposed training bundle.
repair_H119 = F.gelu(repair_linear(
    torch.cat([
        C84,
        Eworld84,
        pool109([], "encode_action", "encoded").expand(len(C84), -1)
    ], dim=-1),
    "z_post.0"
)).numpy().copy()
repair_H119[drug_idx110] = drug_hidden110
repair_ids119 = state_ids.copy()
repair_valid119 = np.asarray(valid84, dtype=bool)
repair_train119 = repair_valid119 & ((repair_ids119 % 2) == 0)
repair_test119 = repair_valid119 & ~repair_train119
repair_targets119 = np.zeros((len(repair_H119), 2), dtype=np.float64)
repair_active119 = np.zeros_like(repair_targets119, dtype=bool)

_good = panel_pos111 >= 0
print("=== RATE ALIGNMENT ===")
for _j, _drug in enumerate(["insulin", "norepinephrine"]):
    _record_rates = panel[_drug].to_numpy()[panel_pos111[_good]]
    print(_drug, "max rate disagreement",
          np.max(np.abs(_record_rates - drug_rates110[_good, _j])))
    _r = np.zeros(len(drug_idx110))
    _r[_good] = np.nan_to_num(
        dose_residuals[_drug][panel_pos111[_good]], nan=0.0
    )
    _a = drug_rates110[:, _j] > 0
    _r[~_a] = 0.0
    repair_targets119[drug_idx110, _j] = _r
    repair_active119[drug_idx110, _j] = _a

# Perturbation differences use the same factual context and never supply causal truth.
repair_D119, repair_Y119, repair_DI119, repair_keys119 = [], [], [], []
for _key, _h in hidden_edits110.items():
    _drug, _factor = _key
    _j = ["insulin", "norepinephrine"].index(_drug)
    _delta = np.asarray(_h, dtype=np.float64) - drug_hidden110
    _target = np.zeros((len(_delta), 2))
    _target[:, _j] = (_factor - 1) * drug_rates110[:, _j]
    repair_D119.append(_delta)
    repair_Y119.append(_target)
    repair_DI119.append(drug_idx110)
    repair_keys119.extend([_key] * len(_delta))
repair_D119 = np.vstack(repair_D119)
repair_Y119 = np.vstack(repair_Y119)
repair_DI119 = np.concatenate(repair_DI119)
repair_dtrain119 = repair_train119[repair_DI119]
repair_dtest119 = repair_test119[repair_DI119]

# Affine factual rows; difference rows have no intercept.
repair_X119 = np.column_stack([repair_H119.astype(np.float64),
                              np.ones(len(repair_H119))])
repair_DX119 = np.column_stack([repair_D119,
                               np.zeros(len(repair_D119))])
repair_decoders119 = {}
repair_reports119 = []
for _weight in [0.25, 1.0, 4.0]:
    _X = np.vstack([repair_X119[repair_train119],
                    np.sqrt(_weight) * repair_DX119[repair_dtrain119]])
    _Y = np.vstack([repair_targets119[repair_train119],
                    np.sqrt(_weight) * repair_Y119[repair_dtrain119]])
    for _alpha in [0.1, 1.0, 10.0]:
        _fit = Ridge(alpha=_alpha, fit_intercept=False).fit(_X, _Y)
        repair_decoders119[(_weight, _alpha)] = _fit
        _pf = _fit.predict(repair_X119)
        _pd = _fit.predict(repair_DX119)
        _row = {"perturb_weight": _weight, "ridge": _alpha}
        for _j, _short in enumerate(["ins", "ne"]):
            _a = repair_test119 & repair_active119[:, _j]
            _non = repair_test119 & ~repair_active119.any(axis=1)
            _d = repair_dtest119 & (repair_Y119[:, _j] != 0)
            _truth = repair_targets119[_a, _j]
            _row[_short+"_innovation_R2"] = (
                1 - np.sum((_pf[_a, _j]-_truth)**2)
                / max(np.sum((_truth-_truth.mean())**2), 1e-12)
            )
            _row[_short+"_no_action_RMS"] = np.sqrt(np.mean(_pf[_non, _j]**2))
            _row[_short+"_edit_slope"] = (
                np.dot(_pd[_d, _j], repair_Y119[_d, _j])
                / np.dot(repair_Y119[_d, _j], repair_Y119[_d, _j])
            )
            _row[_short+"_edit_RMSE"] = np.sqrt(np.mean(
                (_pd[_d, _j]-repair_Y119[_d, _j])**2))
        repair_reports119.append(_row)
print("=== HELD-STAY DECODER CHECK ===")
print(pd.DataFrame(repair_reports119).round(4).to_string(index=False))
print("=== DIRECTION UNITS ===")
print("level std", np.std(levels104, axis=0).round(3))
print("vg coefficient-column differences",
      [float(np.linalg.norm(vg104-direction_all104.coef_[:,192+j])) for j in range(6)])
print("vm coefficient-column differences",
      [float(np.linalg.norm(vm104-direction_all104.coef_[:,192+j])) for j in range(6)])
print("No weights changed:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 116–119 — resolved alignment; decoder test\n"
             "- All 53 unmatched bundles are terminal END-only bundles at 25.0833, zero start rates.\n"
             "- Drug posterior reconstruction RMS 0.00418.\n"
             "- Fitting affine posterior-hidden dose-innovation decoders; validation split is by training stay.\n"
             "- Perturbation features calibrate rate encoding only; causal targets remain record-derived.\n"
             "- No weights changed yet.\n")
```
