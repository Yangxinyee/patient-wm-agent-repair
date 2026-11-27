Latent sampling is the larger problem: the decoder’s valid-row outputs grow from hundredths to units when the prior is sampled. I’ll refit it using sampled head inputs and test constraints that remove latent-sensitive directions. Success would mean small, stable corrections on held-out factual histories; intervention leverage will still need a rollout test.

```python
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

robust_parts = {}
robust_X = {}
robust_state_index = {"first": 4, "extra": 5, "tuning": 3}
robust_samples = 8

for ss0, state_index0 in robust_state_index.items():
    parts0 = cpu_head_parts(decode_meta_sets[ss0], state_files[state_index0])
    robust_parts[ss0] = parts0
    mu0 = parts0["prior"][:, :32]
    sd0 = parts0["prior"][:, 32:].clamp(-10, 5).exp()
    rng0 = torch.Generator().manual_seed(871)
    xs0 = []
    for jj0 in range(robust_samples):
        z0 = mu0 + sd0 * torch.randn(mu0.shape, generator=rng0)
        q0 = ln(parts0["base"] + lin(z0, "z_head") + W["type_emb.weight"][1],
                "query_norm")
        xs0.append((q0 + parts0["additive"]).numpy())
    robust_X[ss0] = np.stack(xs0)
    print(ss0, robust_X[ss0].shape,
          "latent input SD", float(robust_X[ss0].std(0).mean()))

regdirs0 = load_original("last_numeric.weight").numpy()
prior_dirs0 = (load_original("query_norm.weight").numpy()[:, None]
               * load_original("z_head.weight").numpy())
norm_dir0 = load_original("query_norm.weight").numpy()[:, None]

robust_projectors = {}
for ppname0, dirs0 in [
    ("register", regdirs0),
    ("register_prior", np.column_stack([regdirs0, prior_dirs0, norm_dir0]))
]:
    u0, sv0, _ = np.linalg.svd(dirs0, full_matrices=False)
    u0 = u0[:, sv0 > sv0[0]*1e-7]
    robust_projectors[ppname0] = np.eye(192) - u0 @ u0.T

robust_decoders = {}
firstmask0 = np.asarray(valid_glucose_masks["first"], bool)
extra_mask0 = np.asarray(valid_glucose_masks["extra"], bool)
tune_mask0 = np.asarray(valid_glucose_masks["tuning"], bool)
targets0 = {"first": unit_first, "extra": unit_extra, "tuning": unit_tuning}

print("\nLATENT-ROBUST DECODERS")
for projname0, projector0 in robust_projectors.items():
    xtrain0 = robust_X["first"][:, firstmask0].reshape(-1, 192) @ projector0
    ytrain0 = np.tile(np.asarray(unit_first)[firstmask0], robust_samples)
    locr0 = xtrain0.mean(0)
    scaler0 = np.maximum(xtrain0.std(0), 0.05)
    xr0 = (xtrain0-locr0)/scaler0
    for ar0 in [0.1, 10.0, 1000.0]:
        rr0 = Ridge(alpha=ar0).fit(xr0, ytrain0)
        wr0 = projector0 @ (rr0.coef_/scaler0)
        br0 = float(rr0.intercept_ - locr0 @ (rr0.coef_/scaler0))
        robust_decoders[(projname0, ar0)] = (wr0, br0)
        report0 = {}
        for s0, m0 in [("extra", extra_mask0), ("tuning", tune_mask0)]:
            pr0 = robust_X[s0][:, m0] @ wr0 + br0
            yr0 = np.asarray(targets0[s0])[m0]
            report0[s0] = {
                "r2_mean": round(float(r2_score(yr0, pr0.mean(0))), 4),
                "r2_sampled": round(float(r2_score(np.tile(yr0, robust_samples),
                                                  pr0.reshape(-1))), 4),
                "prediction_sd": round(float(pr0.mean(0).std()), 5),
                "latent_sd": round(float(pr0.std(0).mean()), 5),
                "q01_q99": np.round(np.quantile(pr0, [.01, .99]), 5).tolist()
            }
        print(projname0, ar0, report0)

print("\nLATENT-ROBUST HISTORY GATE")
gate_train2 = robust_X["first"].reshape(-1, 192)
gate_loc2 = gate_train2.mean(0)
gate_scale2 = np.maximum(gate_train2.std(0), 0.05)
gate_target2 = np.tile(np.where(firstmask0, 1.0, -1.0), robust_samples)
gate_reg2 = Ridge(alpha=10.0).fit(
    (gate_train2-gate_loc2)/gate_scale2, gate_target2)
gate_w2 = gate_reg2.coef_/gate_scale2
gate_b2 = float(gate_reg2.intercept_ - gate_loc2 @ gate_w2)
for s0, m0 in [("extra", extra_mask0), ("tuning", tune_mask0)]:
    gg0 = robust_X[s0] @ gate_w2 + gate_b2
    print(s0, "valid", short_stats(gg0[:, m0]),
          "invalid", short_stats(gg0[:, ~m0]),
          "error at margin 0.45", float(np.mean((gg0 > .45) != m0[None, :])))
print("CURRENT EDITS", edited())
```