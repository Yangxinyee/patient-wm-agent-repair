The posterior can be edited without changing the rest of the architecture. I’ll first test whether a linear readout of its action input can recover dose across all relevant multipliers, and whether context predicts the usual dose. Accurate dose recovery plus small factual residuals would support a correction that responds strongly off-policy but minimally on the recorded trajectory.

```python
from sklearn.linear_model import Ridge

print("=== SAVED-STATE AND REGISTER SCHEMAS ===")
for rr39_name in ["train_states0", "state_keys_audit", "rr34_rates",
                  "core_probe_indices", "core_probe_h", "core_probe_memory",
                  "core_probe_ages", "core_probe_dt"]:
    rr39_obj = globals().get(rr39_name)
    print(rr39_name, type(rr39_obj), getattr(rr39_obj, "shape", None))
    if hasattr(rr39_obj, "keys"):
        print("keys:", list(rr39_obj.keys()))
print("first state keys:", list(state_keys_audit)[:3])
print("one action bundle:", next((v for v in bundle_records_audit.values()
                                 if any(str(r.get("event", "")) == "insulin"
                                        for r, kind in v)), None))
print("register-related tensors:",
      [(n, SHAPES[n]) for n in NAMES
       if n.startswith(("last_", "type_emb.", "embed.categorical.", "embed_transition."))])

rr39_rates = np.asarray(rr34_rates, dtype=np.float64)
assert rr39_rates.shape == (len(all_u_audit), 2), rr39_rates.shape
rr39_ids = np.asarray([key[0] for key in state_keys_audit])
rr39_X_parts = [all_u_audit.numpy()]
rr39_Y_parts = [np.column_stack([rr39_rates, (rr39_rates > 0).astype(float)])]
rr39_id_parts = [rr39_ids]
for rr39_j, rr39_drug in enumerate(["insulin", "norepinephrine"]):
    rr39_idx = np.flatnonzero(rr39_rates[:, rr39_j] > 0)
    for rr39_factor in [0.0, 0.25, 0.5, 1.5, 2.0, 3.0]:
        rr39_u = rr35_scaled_action_encoding(rr39_idx, rr39_drug, rr39_factor)
        rr39_y = np.column_stack([rr39_rates[rr39_idx],
                                 (rr39_rates[rr39_idx] > 0).astype(float)])
        rr39_y[:, rr39_j] *= rr39_factor
        rr39_X_parts.append(rr39_u.numpy())
        rr39_Y_parts.append(rr39_y)
        rr39_id_parts.append(rr39_ids[rr39_idx])
rr39_X = np.concatenate(rr39_X_parts).astype(np.float64)
rr39_Y = np.concatenate(rr39_Y_parts)
rr39_aug_ids = np.concatenate(rr39_id_parts)
rr39_hold_ids = np.unique(rr39_ids)[::5]
rr39_test = np.isin(rr39_aug_ids, rr39_hold_ids)
rr39_decoder_scores = []
for rr39_alpha in [1e-6, 1e-4, 0.01, 1.0]:
    rr39_fit = Ridge(alpha=rr39_alpha).fit(rr39_X[~rr39_test], rr39_Y[~rr39_test])
    rr39_err = rr39_fit.predict(rr39_X[rr39_test]) - rr39_Y[rr39_test]
    rr39_score = np.sqrt(np.mean(rr39_err ** 2, axis=0))
    rr39_decoder_scores.append((rr39_alpha, rr39_score.tolist()))
print("\n=== ACTION DOSE/PRESENCE DECODER HELD-OUT RMSE ===")
print(rr39_decoder_scores)
rr39_best_alpha = min(rr39_decoder_scores,
                      key=lambda x: x[1][0] + 5*x[1][1] + sum(x[1][2:]))[0]
rr39_action_decoder = Ridge(alpha=rr39_best_alpha).fit(rr39_X, rr39_Y)

rr39_context_X = post_inputs_audit[:, :384].numpy().astype(np.float64)
rr39_context_test = np.isin(rr39_ids, rr39_hold_ids)
rr39_policy_scores = {}
for rr39_j, rr39_drug in enumerate(["insulin", "norepinephrine"]):
    rr39_active = rr39_rates[:, rr39_j] > 0
    rr39_scores = []
    for rr39_alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        rr39_tr = rr39_active & ~rr39_context_test
        rr39_te = rr39_active & rr39_context_test
        rr39_fit = Ridge(alpha=rr39_alpha).fit(rr39_context_X[rr39_tr],
                                             rr39_rates[rr39_tr, rr39_j])
        rr39_err = rr39_fit.predict(rr39_context_X[rr39_te]) - rr39_rates[rr39_te, rr39_j]
        rr39_scores.append((rr39_alpha, float(np.sqrt(np.mean(rr39_err**2)))))
    rr39_policy_scores[rr39_drug] = rr39_scores
print("\n=== ACTIVE-DOSE CONTEXT PREDICTION HELD-OUT RMSE ===")
print(rr39_policy_scores)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 39: direct action-dose decoding and policy centering\n"
            f"- Synthetic multiplier action-readout held-out RMSE: {rr39_decoder_scores}\n"
            f"- Active-dose linear context prediction RMSE: {rr39_policy_scores}\n"
            "- These are small statistical fits, not weight training; no model tensors edited.\n"
            "- Python runs: 39; forward passes: 4.\n")
```