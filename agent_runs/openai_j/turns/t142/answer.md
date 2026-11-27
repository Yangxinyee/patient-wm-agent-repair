The fitted correction strengthens doubled insulin, but does little for halved insulin or norepinephrine. I’ll test it as an insulin-only candidate, not a complete repair. Two replacement units can restrict the correction to glucose; their computed logits must match the fitted correction before I save any weights.

```python
import torch
import torch.nn.functional as F

r129_family = np.asarray(r124_family)
print("Family values/counts:", np.unique(r129_family, return_counts=True))
print("Tilt basis:", r122_basis)

# Identify the feature convention used by the fitted family gate.
r129_gate_trials = {}
for label, X in [("raw", r124_raw), ("standardised", r124_F[:, :-1])]:
    if X.shape[1] == r124_gate.coef_.size:
        pred = r124_gate.predict(X)
        r129_gate_trials[label] = float(np.sqrt(np.mean((pred-r129_family)**2)))
print("Gate reconstruction RMSE:", r129_gate_trials)
r129_gate_layout = min(r129_gate_trials, key=r129_gate_trials.get)

r129_gw = np.asarray(r124_gate.coef_, dtype=np.float64).copy()
r129_gb = float(r124_gate.intercept_)
if r129_gate_layout == "standardised":
    r129_gw /= r124_rs
    r129_gb -= r124_rm @ r129_gw

# Convert the fitted correction to unstandardised input coordinates.
r129_cw = r125_coef[:-1] / r124_rs
r129_cb = float(r125_coef[-1] - r124_rm @ r129_cw)
r129_linear = r124_raw @ r129_cw + r129_cb
for sign in np.unique(r129_family):
    m = r129_family == sign
    print("family", sign, "correction RMS", np.sqrt(np.mean(r129_linear[m]**2)),
          "stored-tilt agreement", np.sqrt(np.mean((r129_linear[m]-r125_tau[m])**2)))

# Build, but do not install, a two-unit gated correction.
r129_gain = 3.0
r129_gate_strength = 10.0
r129_W0 = np.asarray(load_original("attr_heads.value_bin.0.weight")).copy()
r129_b0 = np.asarray(load_original("attr_heads.value_bin.0.bias")).copy()
r129_W2 = r125_compW.copy()
r129_b2 = r125_compb.copy()
j0, j1 = map(int, r125_drop)
r129_W0[j0] = r129_gate_strength*r129_gw + r129_gain*r129_cw
r129_b0[j0] = r129_gate_strength*r129_gb + r129_gain*r129_cb
r129_W0[j1] = r129_gate_strength*r129_gw
r129_b0[j1] = r129_gate_strength*r129_gb
r129_W2[:, j0] = r122_basis
r129_W2[:, j1] = -r122_basis

with torch.no_grad():
    r129_H = F.gelu(F.linear(torch.as_tensor(r124_raw, dtype=torch.float32),
                            torch.as_tensor(r129_W0, dtype=torch.float32),
                            torch.as_tensor(r129_b0, dtype=torch.float32))).numpy()
r129_exact_delta = r129_H @ r129_W2.T + r129_b2 - (
    r122_H @ np.asarray(load_original("attr_heads.value_bin.2.weight")).T
    + np.asarray(load_original("attr_heads.value_bin.2.bias")))
r129_expected_delta = r125_deltaL + r129_gain*r125_tau[:, None]*r122_basis[None, :]
r129_centered_error = r129_exact_delta-r129_expected_delta
r129_centered_error -= r129_centered_error.mean(axis=1, keepdims=True)
print("Candidate/fitted centered-logit RMSE:", np.sqrt(np.mean(r129_centered_error**2)))

r129_P = np.exp(np.log(np.maximum(r122_P, 1e-30)) + r129_exact_delta
                - logsumexp(np.log(np.maximum(r122_P, 1e-30)) + r129_exact_delta,
                            axis=1, keepdims=True))
r129_dNLL = -np.log(r129_P[np.arange(len(r122_bins)), r122_bins]) + np.log(
    r122_P[np.arange(len(r122_bins)), r122_bins])
print("Candidate local dNLL, all/test:",
      np.mean(r129_dNLL), np.mean(r129_dNLL[r122_test]))
print("Active edits:", edited())
```
