Changing activations did not reproduce observation states either. I’ll inspect the embedding helper for a numerical mismatch and separately test the verified drug features. The latter approach is worth pursuing only if held-out changed-dose responses are substantially larger than its unintended factual responses.

```python
import dis

# Test a factual-zero dose-contrast decoder on accurately reconstructed bundles.
phi0 = posterior_phi_repair.detach().numpy().astype(np.float64)
bundle_ids = sm_repair.iloc[posterior_rows_repair].stay_id.to_numpy()
unique_ids = np.sort(np.unique(bundle_ids))
heldout_ids = set(unique_ids[::5])
heldout = np.array([sid in heldout_ids for sid in bundle_ids])
verified = posterior_error_repair < 1e-3

feature_center = phi0[verified & ~heldout].mean(axis=0)
feature_scale = np.maximum(phi0[verified & ~heldout].std(axis=0), 0.05)
xf = np.column_stack(((phi0 - feature_center) / feature_scale,
                      np.ones(len(phi0))))

dose_contrast_blocks = []
for key, cf_phi in posterior_cf_phi_repair.items():
    delta_target = np.asarray(posterior_cf_targets_repair[key], dtype=np.float64)
    active_rows = np.any(delta_target != 0, axis=1)
    delta_features = np.column_stack((
        (np.asarray(cf_phi, dtype=np.float64) - phi0) / feature_scale,
        np.zeros(len(phi0))
    ))
    dose_contrast_blocks.append((key, delta_features, delta_target, active_rows))

fact_train = verified & ~heldout
C0 = xf[fact_train].T @ xf[fact_train] / fact_train.sum()
Cd = np.zeros_like(C0)
Rd = np.zeros((xf.shape[1], 2))
target_scale = np.array([1.5, 0.4])

for key, dx, target, active_rows in dose_contrast_blocks:
    mask = active_rows & verified & ~heldout
    Cd += dx[mask].T @ dx[mask] / mask.sum()
    Rd += dx[mask].T @ (target[mask] / target_scale) / mask.sum()
Cd /= len(dose_contrast_blocks)
Rd /= len(dose_contrast_blocks)

posterior_contrast_fits = {}
contrast_diagnostics = []
for alpha in (0.03, 0.3, 3.0, 30.0, 300.0):
    reg = np.eye(xf.shape[1]) * 0.002
    reg[-1, -1] = 1e-8
    coef = np.linalg.solve(C0 + alpha * Cd + reg, alpha * Rd) * target_scale
    posterior_contrast_fits[alpha] = coef
    test = verified & heldout
    factual_response = xf[test] @ coef
    result = {
        "alpha": alpha, "test_bundles": int(test.sum()),
        "factual_insulin_rms": float(np.sqrt(np.mean(factual_response[:, 0]**2))),
        "factual_ne_rms": float(np.sqrt(np.mean(factual_response[:, 1]**2))),
    }
    for key, dx, target, active_rows in dose_contrast_blocks:
        mask = active_rows & verified & heldout
        col = 0 if key[0] == "insulin" else 1
        prediction = (dx[mask] @ coef)[:, col]
        truth = target[mask, col]
        label = f"{key[0]}_{key[1]}"
        result[label + "_gain"] = float(prediction @ truth / (truth @ truth))
        result[label + "_rmse"] = float(np.sqrt(np.mean((prediction - truth)**2)))
    contrast_diagnostics.append(result)

print("=== HELD-OUT DOSE-CONTRAST DIAGNOSTICS ===")
print(pd.DataFrame(contrast_diagnostics).round(4).to_string(index=False))
print("Verified factual bundles:", int(verified.sum()), "/", len(verified))

print("\n=== ATOM HELPER: NUMERIC IMPLEMENTATION ===")
print("Globals used:", embed_record_atom.__code__.co_names)
print(dis.Bytecode(embed_record_atom).dis()[-6200:])
```
