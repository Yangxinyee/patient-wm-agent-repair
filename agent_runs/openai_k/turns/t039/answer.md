Including the current world bundle in memory reduced the summary-state error to about 0.000018, a substantial improvement. I’ll verify that convention on all 64 probes, then test whether posterior features can express conditionally centered dose changes. Accurate reconstruction plus a useful dose signal would support a low-rank edit; weak signal recovery would count against it.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

def repaired_core_local(h, e, u, z, memory, mask, ages, include_current=True):
    if include_current:
        memory = torch.cat([memory, e[:, None, :]], dim=1)
        mask = torch.cat([mask, torch.ones(len(h), 1, dtype=torch.bool)], dim=1)
        ages = torch.cat([ages, torch.zeros(len(h), 1, 1)], dim=1)
    tokens = torch.cat([
        h + W0["core.slots"][None], e[:, None], u[:, None],
        lin0(z, "z_in")[:, None]
    ], dim=1)
    tokens = audit_block(tokens, "core.prelude.0")
    E = norm0(tokens[:, :4], "core.prelude_norm")
    mem = memory + mlp0(torch.log1p(ages), "core.age", "gelu")
    state = audit_block(h + E, "core.blocks.0.0", mem, mask)
    state = audit_block(state, "core.blocks.0.1", mem, mask)
    delta = F.softplus(W0["core.delta_in"])
    retention = torch.exp(-delta * torch.exp(W0["core.a_in"]))
    new_h = retention[None, None] * h + delta[None, None] * lin0(E, "core.b_in")
    new_h = new_h + state - (h + E)
    new_h = audit_block(lin0(new_h, "core.c_out"), "core.coda.0")
    return norm0(new_h.mean(1), "core.out_norm"), new_h

rr34_out, rr34_h = repaired_core_local(
    core_probe_h, core_probe_e, core_probe_u, core_probe_z,
    core_probe_memory, core_probe_mask, core_probe_ages
)
rr34_target = torch.as_tensor(sx[np.asarray(core_probe_indices, dtype=int)])
rr34_out_err = float((rr34_out - rr34_target[:, :192]).square().mean().sqrt())
rr34_h_err = float(
    (rr34_h - rr34_target[:, 192:960].reshape(-1, 4, 192)).square().mean().sqrt()
)
print("=== FULL PROBE VALIDATION ===")
print("out RMS:", rr34_out_err, "h RMS:", rr34_h_err)
print("max out error:", float((rr34_out - rr34_target[:, :192]).abs().max()))

rr34_H = post_hidden_audit.detach().cpu().numpy().astype(np.float64)
rr34_rates = np.zeros((len(rr34_H), 2))
rr34_residual = np.zeros((len(rr34_H), 2))
rr34_rows = np.asarray(state_drug_rows0, dtype=int)
rr34_rates[rr34_rows] = state_drug_rate0
rr34_residual[rr34_rows] = state_drug_innov0 * (state_drug_rate0 > 0)
rr34_known = np.zeros(len(rr34_H), dtype=bool)
rr34_known[rr34_rows] = True
rr34_fold = np.asarray([k[0] % 2 for k in state_keys_audit])
rr34_cv = []

for rr34_alpha in (.001, .01, .1, 1., 10., 100.):
    rr34_pred = np.zeros_like(rr34_residual)
    for rr34_f in (0, 1):
        rr34_tr = rr34_fold != rr34_f
        rr34_te = ~rr34_tr
        rr34_fit = Ridge(alpha=rr34_alpha).fit(
            rr34_H[rr34_tr], rr34_residual[rr34_tr]
        )
        rr34_pred[rr34_te] = rr34_fit.predict(rr34_H[rr34_te])
    rr34_r2 = []
    for rr34_j in range(2):
        rr34_active = rr34_known & (rr34_rates[:, rr34_j] > 0)
        rr34_truth = rr34_residual[rr34_active, rr34_j]
        rr34_r2.append(1 - np.mean(
            (rr34_pred[rr34_active, rr34_j] - rr34_truth)**2
        ) / np.var(rr34_truth))
    rr34_zero = ~rr34_known
    rr34_cv.append((float(np.mean(rr34_r2)), rr34_alpha, rr34_r2,
                    np.sqrt(np.mean(rr34_pred[rr34_zero]**2, axis=0)).tolist()))

print("\n=== DOSE-RESIDUAL FEATURE RECOVERY ===")
for rr34_row in sorted(rr34_cv, reverse=True):
    print("mean R2, alpha, active R2, non-action RMS:", rr34_row)

rr34_best_alpha = max(rr34_cv)[1]
dose_feature_fit = Ridge(alpha=rr34_best_alpha).fit(rr34_H, rr34_residual)
dose_feature_coef = torch.as_tensor(dose_feature_fit.coef_.T, dtype=torch.float32)
dose_feature_bias = torch.as_tensor(dose_feature_fit.intercept_, dtype=torch.float32)
print("selected alpha:", rr34_best_alpha)
print("feature coefficient norms:", dose_feature_coef.norm(dim=0).tolist())

print("\n=== HEAD/REGISTER VOCABULARY ===")
print("vocab keys:", list(VOCAB))
for rr34_key in ("codes", "types", "kinds", "event_kinds", "sizes"):
    if rr34_key in VOCAB:
        print(rr34_key, str(VOCAB[rr34_key])[:1500])
print("last_numeric column norms:", W0["last_numeric.weight"].norm(dim=0).tolist())

with open("/work/notes.md", "a") as rr34_f:
    rr34_f.write("\n## Run 34: recurrent reconstruction resolved\n"
                 "- The current WORLD bundle is appended to memory BEFORE cross-attention.\n"
                 "- Age uses log1p(seconds), GELU; age embedding is added to both key and value inputs.\n"
                 "- General validated helper: repaired_core_local(h,e,u,z,prior_memory,mask,ages).\n"
                 f"- Full 64-probe RMS: out={rr34_out_err:.8g}, h={rr34_h_err:.8g}.\n"
                 "- Candidate posterior dose features are dose_feature_coef (192x2) and dose_feature_bias (2), from original post_hidden_audit.\n"
                 "- Cross-validation results: " + repr(sorted(rr34_cv, reverse=True)) + "\n"
                 "- These are diagnostic statistical fits, not edits; next test off-policy feature responses and choose physiological transfer directions.\n"
                 "- Python runs: 34; forward passes: 4; no weights edited.\n")
```