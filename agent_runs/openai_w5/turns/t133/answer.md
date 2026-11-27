The candidate preserves ordinary encoder outputs closely, but its rate errors are still too large. The problem appears to be compensating for layer normalization, rather than recovering rates initially. I’ll try normalized output targets with analytic rate arithmetic. This would be supported by much smaller rate errors without a substantial increase in ordinary-output error.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from pathlib import Path

rate_feature118 = np.zeros((2, 768), dtype=np.float64)
for d in range(2):
    j0, j1, j2, j3 = neurons117[4*d:4*d+4]
    rate_feature118[d, [j0, j1, j2, j3]] = [
        1, -1, np.sqrt(2)-1, -(np.sqrt(2)-1)
    ]
rate_direct118 = hidden_newv117.astype(np.float64) @ rate_feature118.T * rate_scale105
print("analytic hidden-rate RMSE:",
      np.sqrt(np.mean((rate_direct118-qv117)**2, axis=0)).round(7))

normal_raw118 = (
    raw117.astype(np.float64) - raw117.astype(np.float64).mean(1, keepdims=True)
) / np.sqrt(raw117.astype(np.float64).var(1, keepdims=True) + 1e-5)
target_norm118 = channel_raw_target117(q117, normal_raw118)
gamma118 = enc_orig117["encode_action.norm.weight"].clone()
bias118 = enc_orig117["encode_action.norm.bias"].clone()
gamma118[R117] = 1
bias118[R117] = 0
norm_trials118 = []
for alpha in [0.001, 0.01, 0.1]:
    fit = Ridge(alpha=alpha).fit(xs117, target_norm118)
    coef = fit.coef_ / xsd117[None, :]
    intercept = fit.intercept_ - coef @ xmean117
    for d in range(2):
        i, j = R117[2*d:2*d+2]
        common = (coef[i] + coef[j]) / 2
        common_bias = (intercept[i] + intercept[j]) / 2
        coef[i] = common + enc_delta117 * rate_feature118[d]
        coef[j] = common - enc_delta117 * rate_feature118[d]
        intercept[i] = common_bias
        intercept[j] = common_bias
    raw_pred = hidden_newv117.astype(np.float64) @ coef.T + intercept
    u = F.layer_norm(torch.tensor(raw_pred, dtype=torch.float32),
                     (192,), gamma118, bias118, 1e-5).numpy()
    decoded = np.column_stack([
        u[:, R117[0]]-u[:, R117[1]],
        u[:, R117[2]]-u[:, R117[3]]
    ]) / (2*enc_delta117) * rate_scale105
    err = decoded - qv117
    typical = (qv117[:, 0] <= 6) & (qv117[:, 1] <= 1)
    active = pv117.any(1)
    report = {
        "alpha": alpha,
        "rate_rmse_I_NE": np.sqrt(np.mean(err**2, axis=0)).round(6).tolist(),
        "typical_rate_rmse_I_NE": np.sqrt(np.mean(err[typical]**2, axis=0)).round(6).tolist(),
        "max_rate_error_I_NE": np.max(np.abs(err), axis=0).round(5).tolist(),
        "ordinary_u_rmse": float(np.sqrt(np.mean((u[:, K117]-outv117[:, K117])**2))),
        "raw_sigma_min_max": np.sqrt(raw_pred.var(1))[active].min().round(5).item().__str__()
                              + " / " + str(np.sqrt(raw_pred.var(1))[active].max().round(5))
    }
    norm_trials118.append((report["ordinary_u_rmse"] +
                          np.mean(np.mean((err/rate_scale105)**2, axis=0)),
                          coef, intercept, report))
    print(report)
best_norm118 = min(norm_trials118, key=lambda x: x[0])
enc_candidate118 = {n: v.clone() for n, v in enc_candidate117.items()}
enc_candidate118["encode_action.mlp2.2.weight"] = torch.tensor(best_norm118[1], dtype=torch.float32)
enc_candidate118["encode_action.mlp2.2.bias"] = torch.tensor(best_norm118[2], dtype=torch.float32)
print("No new edits installed:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 118: normalized analytic rate channel\n"
            "- Tested exact GELU-pair rate arithmetic with normalized-output distillation to avoid q times unknown layer-normalization scale.\n"
            "- Best local report: " + repr(best_norm118[3]) + "\n"
            "- No encoder candidate installed yet.\n")
```
