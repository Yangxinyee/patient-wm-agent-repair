The candidate failed the equivalence check because its gate leaked correction into MAP predictions. The insulin fit was unconstrained there, so those values can be large. I’ll verify the gate’s signed labels and strengthen its separation; negligible logit error and restored factual loss would confirm the fix.

```python
r130_signed_family = 2*r124_family.astype(float)-1
r130_gate_checks = {}
for label, X in [("raw", r124_raw), ("standardised", r124_F[:, :-1])]:
    p = r124_gate.predict(X)
    r130_gate_checks[label] = float(np.sqrt(np.mean((p-r130_signed_family)**2)))
print("Signed-family gate RMSE:", r130_gate_checks)
r130_layout = min(r130_gate_checks, key=r130_gate_checks.get)
r129_gw = np.asarray(r124_gate.coef_, dtype=np.float64).copy()
r129_gb = float(r124_gate.intercept_)
if r130_layout == "standardised":
    r129_gw /= r124_rs
    r129_gb -= r124_rm @ r129_gw
r130_g = r124_raw @ r129_gw + r129_gb
print("Gate ranges, glucose/MAP:",
      (r130_g[r124_family].min(), r130_g[r124_family].max()),
      (r130_g[~r124_family].min(), r130_g[~r124_family].max()))
print("Ungated MAP correction range:",
      r129_linear[~r124_family].min(), r129_linear[~r124_family].max())

r129_gate_strength = max(
    40.0,
    (r129_gain*np.max(np.abs(r129_linear[~r124_family]))+12.0)
    / np.min(np.abs(r130_g[~r124_family]))
)
r129_W0[j0] = r129_gate_strength*r129_gw + r129_gain*r129_cw
r129_b0[j0] = r129_gate_strength*r129_gb + r129_gain*r129_cb
r129_W0[j1] = r129_gate_strength*r129_gw
r129_b0[j1] = r129_gate_strength*r129_gb

with torch.no_grad():
    r130_H = F.gelu(F.linear(torch.as_tensor(r124_raw, dtype=torch.float32),
                            torch.as_tensor(r129_W0, dtype=torch.float32),
                            torch.as_tensor(r129_b0, dtype=torch.float32))).numpy()
r130_delta = r130_H @ r129_W2.T + r129_b2 - (
    r122_H @ np.asarray(load_original("attr_heads.value_bin.2.weight")).T
    + np.asarray(load_original("attr_heads.value_bin.2.bias")))
r130_err = r130_delta-r129_expected_delta
r130_err -= r130_err.mean(axis=1, keepdims=True)
r130_L = np.log(np.maximum(r122_P, 1e-30)) + r130_delta
r130_P = np.exp(r130_L-logsumexp(r130_L, axis=1, keepdims=True))
r130_dNLL = -np.log(r130_P[np.arange(len(r122_bins)), r122_bins]) + np.log(
    r122_P[np.arange(len(r122_bins)), r122_bins])

print("Gate strength:", r129_gate_strength)
print("Centered-logit error RMS/max:",
      np.sqrt(np.mean(r130_err**2)), np.max(np.abs(r130_err)))
print("Local factual dNLL, all/test:",
      np.mean(r130_dNLL), np.mean(r130_dNLL[r122_test]))
for key, arr in r124_CF.items():
    print("Counterfactual features:", key, type(arr).__name__, getattr(arr, "shape", None))
print("Active edits:", edited())
```
