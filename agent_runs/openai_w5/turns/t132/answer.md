Freeing the attention head increased factual value NLL by 0.00246, leaving about half the allowed margin. I’ll now build and test the rate-channel encoder locally, without installing it. It must recover changed rates accurately while preserving the encoder’s ordinary outputs; otherwise the remaining factual budget is unlikely to cover it.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from pathlib import Path

R117 = np.asarray(reserved103, dtype=int)
K117 = np.setdiff1d(np.arange(192), R117)
enc_delta117 = 0.003
enc_carrier117 = 0.01
enc_orig117 = {n: load_original(n).clone() for n in NAMES if n.startswith("encode_action.")}

def enc_set117(q, p, weights):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=bool)
    n = p.sum(1)
    aggregate = torch.zeros((len(q), 192), dtype=torch.float32)
    for d, drug in enumerate(["insulin", "norepinephrine"]):
        ii = np.flatnonzero(p[:, d])
        if len(ii):
            a = action_atom_exact(drug, q[ii, d])
            hidden = F.gelu(F.linear(a, weights["encode_action.mlp1.0.weight"],
                                      weights["encode_action.mlp1.0.bias"]))
            atom_out = F.linear(hidden, weights["encode_action.mlp1.2.weight"],
                                        weights["encode_action.mlp1.2.bias"])
            aggregate[ii] += atom_out
    aggregate /= torch.tensor(np.sqrt(np.maximum(n, 1)), dtype=torch.float32)[:, None]
    size = torch.tensor(np.log1p(n), dtype=torch.float32)[:, None]
    mixed = aggregate + F.linear(size, weights["encode_action.size.weight"],
                                       weights["encode_action.size.bias"])
    hidden2 = F.gelu(F.linear(mixed, weights["encode_action.mlp2.0.weight"],
                                    weights["encode_action.mlp2.0.bias"]))
    raw = F.linear(hidden2, weights["encode_action.mlp2.2.weight"],
                           weights["encode_action.mlp2.2.bias"])
    out = F.layer_norm(raw, (192,), weights["encode_action.norm.weight"],
                                   weights["encode_action.norm.bias"], 1e-5)
    return mixed.numpy(), hidden2.numpy(), raw.numpy(), out.numpy()

rng117 = np.random.default_rng(117)
def channel_dataset117(n_synthetic, n_record):
    config = rng117.integers(0, 4, n_synthetic)
    p = np.column_stack([config & 1, (config >> 1) & 1]).astype(bool)
    q = rng117.uniform(0, 1, (n_synthetic, 2)) * caps105
    q[~p] = 0
    recorded = panel[["insulin", "norepinephrine"]].fillna(0).to_numpy()
    qr = recorded[rng117.integers(0, len(recorded), n_record)]
    qr = np.minimum(qr, caps105)
    return np.concatenate([q, qr]), np.concatenate([p, qr > 0])

q117, p117 = channel_dataset117(5200, 3000)
qv117, pv117 = channel_dataset117(2000, 1200)
mix117, hid117, raw117, out117 = enc_set117(q117, p117, enc_orig117)
mixv117, hidv117, rawv117, outv117 = enc_set117(qv117, pv117, enc_orig117)

restore117 = Ridge(alpha=0.01).fit(mix117[:, K117].astype(np.float64),
                                  mix117[:, R117].astype(np.float64))
restore_error117 = restore117.predict(mixv117[:, K117]) - mixv117[:, R117]
print("original private mixed-coordinate reconstruction RMSE:",
      np.sqrt(np.mean(restore_error117**2, axis=0)).round(5))

enc_candidate117 = {n: v.clone() for n, v in enc_orig117.items()}
enc_candidate117["encode_action.mlp1.2.weight"][R117] = torch.tensor(
    enc_carrier117 * first_rate116.coef_, dtype=torch.float32)
enc_candidate117["encode_action.mlp1.2.bias"][R117] = torch.tensor(
    enc_carrier117 * first_rate116.intercept_, dtype=torch.float32)
enc_candidate117["encode_action.size.weight"][R117] = 0
enc_candidate117["encode_action.size.bias"][R117] = 0

w20 = enc_orig117["encode_action.mlp2.0.weight"].numpy().astype(np.float64)
b20 = enc_orig117["encode_action.mlp2.0.bias"].numpy().astype(np.float64)
new_w20 = w20.copy()
new_w20[:, K117] += w20[:, R117] @ restore117.coef_
new_w20[:, R117] = 0
new_b20 = b20 + w20[:, R117] @ restore117.intercept_

importance117 = np.linalg.norm(
    enc_orig117["encode_action.mlp2.2.weight"].numpy()[K117], axis=0)
neurons117 = np.argsort(importance117)[:8]
gate117 = 200.0
threshold117 = (1 + np.sqrt(2)) / 2
for d in range(2):
    j0, j1, j2, j3 = neurons117[4*d:4*d+4]
    new_w20[[j0, j1, j2, j3]] = 0
    new_b20[[j0, j1, j2, j3]] = 0
    new_w20[j0, R117[d]] = 1 / enc_carrier117
    new_w20[j1, R117[d]] = -1 / enc_carrier117
    new_w20[j2, R117[d]] = 1 / enc_carrier117
    for j in [j2, j3]:
        new_w20[j, R117[2:]] = gate117 / enc_carrier117
        new_b20[j] = -gate117 * threshold117
enc_candidate117["encode_action.mlp2.0.weight"] = torch.tensor(new_w20, dtype=torch.float32)
enc_candidate117["encode_action.mlp2.0.bias"] = torch.tensor(new_b20, dtype=torch.float32)

_, hidden_new117, _, _ = enc_set117(q117, p117, enc_candidate117)
_, hidden_newv117, _, _ = enc_set117(qv117, pv117, enc_candidate117)

def channel_raw_target117(q, raw):
    raw = raw.astype(np.float64)
    sigma = np.sqrt(raw.var(1) + 1e-5)
    rates = q / rate_scale105
    m = raw[:, R117].mean(1)
    remaining_energy = np.sum((raw[:, R117] - m[:, None])**2, axis=1)
    rate_terms = enc_delta117 * rates * sigma[:, None]
    spare = np.sqrt(np.maximum(remaining_energy -
                    2*np.sum(rate_terms**2, axis=1), 0) / 4)
    target = raw.copy()
    target[:, R117[0]] = m + spare + rate_terms[:, 0]
    target[:, R117[1]] = m + spare - rate_terms[:, 0]
    target[:, R117[2]] = m - spare + rate_terms[:, 1]
    target[:, R117[3]] = m - spare - rate_terms[:, 1]
    return target

target_raw117 = channel_raw_target117(q117, raw117)
xmean117 = hidden_new117.mean(0).astype(np.float64)
xsd117 = np.maximum(hidden_new117.std(0).astype(np.float64), 0.05)
xs117 = (hidden_new117.astype(np.float64) - xmean117) / xsd117
xsv117 = (hidden_newv117.astype(np.float64) - xmean117) / xsd117
encoder_trials117 = []
for alpha in [0.001, 0.01, 0.1]:
    fit = Ridge(alpha=alpha).fit(xs117, target_raw117)
    predicted = torch.tensor(fit.predict(xsv117), dtype=torch.float32)
    gamma = enc_orig117["encode_action.norm.weight"].clone()
    bias = enc_orig117["encode_action.norm.bias"].clone()
    gamma[R117] = 1
    bias[R117] = 0
    u = F.layer_norm(predicted, (192,), gamma, bias, 1e-5).numpy()
    decoded = np.column_stack([
        u[:, R117[0]] - u[:, R117[1]],
        u[:, R117[2]] - u[:, R117[3]]
    ]) / (2 * enc_delta117) * rate_scale105
    err = decoded - qv117
    report = {
        "alpha": alpha,
        "rate_rmse_I_NE": np.sqrt(np.mean(err**2, axis=0)).round(6).tolist(),
        "rate_max_error_I_NE": np.max(np.abs(err), axis=0).round(5).tolist(),
        "ordinary_u_rmse": float(np.sqrt(np.mean((u[:, K117]-outv117[:, K117])**2))),
        "absent_rate_rmse": np.sqrt(np.mean(err[~pv117.any(1)]**2, axis=0)).round(6).tolist()
    }
    encoder_trials117.append((report["ordinary_u_rmse"] +
                             np.mean(np.mean((err/rate_scale105)**2, axis=0)), fit, report))
    print(report)

best_encoder117 = min(encoder_trials117, key=lambda x: x[0])
fit117 = best_encoder117[1]
coef117 = fit117.coef_ / xsd117[None, :]
intercept117 = fit117.intercept_ - coef117 @ xmean117
enc_candidate117["encode_action.mlp2.2.weight"] = torch.tensor(coef117, dtype=torch.float32)
enc_candidate117["encode_action.mlp2.2.bias"] = torch.tensor(intercept117, dtype=torch.float32)
enc_candidate117["encode_action.norm.weight"][R117] = 1
enc_candidate117["encode_action.norm.bias"][R117] = 0
print("No encoder tensors installed; current edits:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward 9 / run 117\n"
            "- Freeing prelude head 3 costs +0.00246 factual value NLL on tuning A (1.64179 versus 1.63933).\n"
            "- Prepared an analytic first-hidden rate channel and linear encoder distillation; not installed.\n"
            "- Encoder validation report: " + repr(best_encoder117[2]) + "\n")
```
