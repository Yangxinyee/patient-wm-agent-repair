The action input recovers changed doses reasonably well, but a linear context model predicts norepinephrine poorly. Using that center could disturb factual predictions. I’ll try the posterior’s existing nonlinear features, penalizing any change in the center when dose changes; success means recovering factual dose innovations without subtracting the intended intervention signal.

```python
import torch.nn.functional as tF

rr41_n = len(all_u_audit)
rr41_cp = np.asarray(causal_positions_audit, dtype=int)
rr41_innov = np.zeros((rr41_n, 2), dtype=np.float64)
rr41_valid = (rr41_cp >= 0) & (rr41_cp < len(dose_resid))
rr41_innov[rr41_valid] = dose_resid[rr41_cp[rr41_valid]]
rr41_innov *= (rr39_rates > 0)
print("Innovation targets: active SD",
      [float(np.std(rr41_innov[rr39_rates[:, j] > 0, j])) for j in range(2)])
print("Existing innovation-target object:",
      type(globals().get("innov_targets_audit")),
      getattr(globals().get("innov_targets_audit"), "shape", None))

rr41_index_parts = [np.arange(rr41_n)]
for rr41_j in range(2):
    rr41_idx = np.flatnonzero(rr39_rates[:, rr41_j] > 0)
    for rr41_factor in [0.0, 0.25, 0.5, 1.5, 2.0, 3.0]:
        rr41_index_parts.append(rr41_idx)
rr41_orig_idx = np.concatenate(rr41_index_parts)
assert len(rr41_orig_idx) == len(rr39_X)
rr41_aug_inputs = torch.cat([
    post_inputs_audit[rr41_orig_idx, :384],
    torch.tensor(rr39_X, dtype=torch.float32)
], dim=1)
rr41_H = tF.gelu(tF.linear(rr41_aug_inputs,
                          W0["z_post.0.weight"], W0["z_post.0.bias"])).numpy().astype(np.float64)
rr41_D = rr39_action_decoder.predict(rr39_X)[:, :2]
rr41_gate = (rr39_action_decoder.predict(rr39_X)[:, 2:] > 0.5)
rr41_D *= rr41_gate
rr41_expected = rr41_D[:rr41_n] - rr41_innov
rr41_dH = rr41_H[rr41_n:] - rr41_H[rr41_orig_idx[rr41_n:]]
rr41_train = ~rr39_context_test
rr41_aug_train = ~np.isin(rr39_aug_ids[rr41_n:], rr39_hold_ids)
rr41_center_models = {}
rr41_center_scores = []
for rr41_penalty in [0.0, 0.01, 0.1, 1.0, 10.0]:
    rr41_fit_X = np.concatenate([
        np.column_stack([rr41_H[:rr41_n][rr41_train], np.ones(rr41_train.sum())]),
        np.sqrt(rr41_penalty) * np.column_stack([
            rr41_dH[rr41_aug_train], np.zeros(rr41_aug_train.sum())])
    ])
    rr41_fit_Y = np.concatenate([
        rr41_expected[rr41_train],
        np.zeros((rr41_aug_train.sum(), 2))
    ])
    rr41_fit = Ridge(alpha=1.0, fit_intercept=False).fit(rr41_fit_X, rr41_fit_Y)
    rr41_center_models[rr41_penalty] = rr41_fit
    rr41_phi = rr41_D - rr41_fit.predict(
        np.column_stack([rr41_H, np.ones(len(rr41_H))]))
    rr41_rmse = []
    rr41_background = []
    rr41_slopes = []
    for rr41_j in range(2):
        rr41_active_test = rr39_context_test & (rr39_rates[:, rr41_j] > 0)
        rr41_rmse.append(float(np.sqrt(np.mean(
            (rr41_phi[:rr41_n][rr41_active_test, rr41_j]
             - rr41_innov[rr41_active_test, rr41_j])**2))))
        rr41_absent_test = rr39_context_test & (rr39_rates[:, rr41_j] == 0)
        rr41_background.append(float(np.sqrt(np.mean(
            rr41_phi[:rr41_n][rr41_absent_test, rr41_j]**2))))
        rr41_true_delta = rr39_Y[rr41_n:, rr41_j] - rr39_rates[rr41_orig_idx[rr41_n:], rr41_j]
        rr41_phi_delta = rr41_phi[rr41_n:, rr41_j] - rr41_phi[rr41_orig_idx[rr41_n:], rr41_j]
        rr41_own = (np.abs(rr41_true_delta) > 1e-8) & ~rr41_aug_train
        rr41_slopes.append(float(np.dot(rr41_true_delta[rr41_own],
                                       rr41_phi_delta[rr41_own])
                                 / np.dot(rr41_true_delta[rr41_own],
                                          rr41_true_delta[rr41_own])))
    rr41_center_scores.append({"invariance_penalty": rr41_penalty,
                              "active_innovation_RMSE": rr41_rmse,
                              "absent_drug_RMS": rr41_background,
                              "dose_response_slopes": rr41_slopes})
print("\n=== NONLINEAR CENTER WITH DOSE-INVARIANCE CONSTRAINT ===")
for rr41_score in rr41_center_scores:
    print(rr41_score)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 41: nonlinear, dose-invariant policy centering\n"
            "- Linear context centering is inadequate, especially for norepinephrine; testing existing nonlinear posterior features with counterfactual invariance constraints.\n"
            f"- Diagnostics: {rr41_center_scores}\n"
            "- No tensors edited. Python runs: 41; forward passes: 4.\n")
```