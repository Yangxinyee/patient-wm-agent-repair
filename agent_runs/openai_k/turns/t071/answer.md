The posterior-response directions require large latent changes, which risks saturation and factual damage. I’ll test the prior-to-head path instead: it may offer stronger dose sensitivity without altering the recurrent update. Consistent, moderate response directions would support that approach; weak or erratic responses would refute it.

```python
t65 = time.time()
print("=== STATE DOSE-RECOVERY RESULTS ===")
print(repr(state_recovery0)[:1700])
print("\n=== RETAINED REGRESSION MODELS ===")
for name, obj in list(globals().items()):
    if name.startswith(("rr39", "rr41")) and hasattr(obj, "coef_"):
        print(name, type(obj).__name__, np.asarray(obj.coef_).shape,
              "intercept:", np.asarray(obj.intercept_).tolist())
for key, obj in rr41_center_models.items():
    print("centre", key, type(obj).__name__,
          (np.asarray(obj.coef_).shape if hasattr(obj, "coef_") else
           repr(obj)[:180]))

print("\n=== PRIOR TENSORS ===")
for name in NAMES:
    if "z_prior" in name:
        print(name, SHAPES[name])

probe65 = np.unique(np.concatenate([
    c["indices"][::6] for c in CF59 if c["factor"] == 2.
]))
o65, h65 = r59_core(probe65)
base65 = r59_features(o65, h65, probe65)[0]
J65 = np.zeros((len(probe65), 4, 32), dtype=np.float64)
bias_key65 = "z_prior.2.bias"
backup65 = W0[bias_key65].clone()
eps65 = 0.05
try:
    for j in range(32):
        delta = torch.zeros_like(backup65)
        delta[j] = eps65
        W0[bias_key65] = backup65 + delta
        plus = r59_features(o65, h65, probe65)[0]
        W0[bias_key65] = backup65 - delta
        minus = r59_features(o65, h65, probe65)[0]
        J65[:, :, j] = (plus - minus) / (2 * eps65)
finally:
    W0[bias_key65] = backup65

A65 = (J65 / scale56[None, :, None]).reshape(-1, 32)
Y65 = np.broadcast_to(target56 / scale56[:, None],
                     (len(probe65), 4, 2)).reshape(-1, 2)
D65_candidates = {}
for penalty in [1e-5, 1e-4, 0.001, 0.01, 0.1, 1.]:
    direction = np.linalg.solve(
        A65.T @ A65 / len(A65) + penalty * np.eye(32),
        A65.T @ Y65 / len(A65))
    D65_candidates[penalty] = direction
    response = np.einsum("nsz,zd->nsd", J65, direction)
    print("penalty", penalty, "direction norms",
          np.round(np.linalg.norm(direction, axis=0), 3).tolist(),
          "mean response", np.round(response.mean(0), 3).tolist(),
          "response SD", np.round(response.std(0), 3).tolist())

print("Seconds:", round(time.time() - t65, 1), "Edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 65: prior-head response assay\n"
            "- Posterior response directions are weak and require large changes; testing prior-to-head route instead.\n"
            "- Measured numerical prior-mean sensitivities over mixed insulin/NE probes and fitted analytical response directions.\n"
            "- J65 and D65_candidates retained; all scratch weights restored. No submitted edits.\n"
            "- Python runs: 65; forward passes: 4.\n")
```