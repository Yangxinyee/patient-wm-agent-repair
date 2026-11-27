The probability capture produced no usable data, so it tells us nothing about the proposed route. I’ll remove the restrictive shape filter and verify that captured probabilities reproduce the helper’s predicted means before computing derivatives. Only a successful reconstruction will justify using the resulting route.

```python
def r267_full_pre(pre, ids):
    h = audit_block(linear228(pre, "core.c_out"), "core.coda.0")
    out = norm0(h.mean(1), "core.out_norm")
    captured = []
    old_softmax = torch.softmax

    def capture(x, dim, *args, **kwargs):
        p = old_softmax(x, dim, *args, **kwargs)
        if p.ndim >= 2 and p.shape[-1] == 17:
            captured.append(np242(p).copy())
        return p

    torch.softmax = capture
    try:
        means = r49_readouts(out, h, ids)
    finally:
        torch.softmax = old_softmax

    n = len(ids)
    reduced = []
    for p in captured:
        if p.shape[0] == n:
            reduced.append(p.reshape(n, -1, 17).mean(axis=1))
        elif p.ndim >= 3 and p.shape[-2] == n:
            axes = tuple(range(p.ndim - 2))
            reduced.append(p.mean(axis=axes))
        else:
            raise ValueError(("Unresolved probability layout", p.shape, n))
    return np242(means), reduced, [p.shape for p in captured]

r268_means, r268_captured, r268_shapes = r267_full_pre(r267_pre, r267_ids)
print("Raw probability shapes:", r268_shapes)
assert len(r268_captured) == 4, len(r268_captured)
r267_P = np.stack(r268_captured, axis=1).astype(np.float64)
r267_centres = np242(r49_centres).astype(np.float64)
assert r267_centres.shape == (4, 17)
r268_reconstructed = (r267_P * r267_centres[None]).sum(axis=-1)
r268_error = np.max(np.abs(r268_reconstructed - r268_means))
print("Maximum reconstructed-mean discrepancy:", r268_error)
assert r268_error < 0.001, "Do not fit a route with an unverified probability layout."

t268 = time.time()
r267_eps = 0.025
r267_Jlog = np.empty((len(r267_sel), 4, 17, 192), dtype=np.float64)
for j in range(192):
    perturb = torch.zeros((1, 1, 192), dtype=torch.float32)
    perturb[..., j] = r267_eps
    _, pp, _ = r267_full_pre(r267_pre + perturb, r267_ids)
    _, pm, _ = r267_full_pre(r267_pre - perturb, r267_ids)
    pp = np.stack(pp, axis=1).astype(np.float64)
    pm = np.stack(pm, axis=1).astype(np.float64)
    r267_Jlog[..., j] = (
        np.log(np.maximum(pp, 1e-12))
        - np.log(np.maximum(pm, 1e-12))
    ) / (2 * r267_eps)

r267_Jmean = np.einsum(
    "ncb,ncbi,cb->nci", r267_P, r267_Jlog, r267_centres
)
r267_sensor_weights = np.array([1538, 488, 1109, 1616], dtype=float)
r267_sensor_weights /= r267_sensor_weights.sum()
r267_Fisher = np.einsum(
    "ncb,ncbi,ncbj,c->ij",
    r267_P, r267_Jlog, r267_Jlog, r267_sensor_weights,
    optimize=True
) / len(r267_sel)
r267_A = r267_Jmean.mean(axis=0)
r267_desired = np.array([0., 0., 1., 1.])
r267_scale = np.trace(r267_Fisher) / 192
r267_routes = {}

old_direction268 = np.asarray(rr_Ndirection)
old_response268 = r267_A @ old_direction268
print("Existing MAP direction mean response:", np.round(old_response268, 5))
print("Existing direction Fisher cost per squared arterial response:",
      float(old_direction268 @ r267_Fisher @ old_direction268)
      / max(old_response268[2] ** 2, 1e-12))

for ridge in [0.001, 0.01, 0.1, 1.0]:
    metric = r267_Fisher + ridge * r267_scale * np.eye(192)
    U = np.linalg.solve(metric, r267_A.T)
    direction = U @ np.linalg.pinv(r267_A @ U, rcond=1e-9) @ r267_desired
    r267_routes[ridge] = direction
    responses = np.einsum("nci,i->nc", r267_Jmean, direction)
    print("ridge", ridge, "norm", round(float(np.linalg.norm(direction)), 4),
          "mean", np.round(responses.mean(0), 4),
          "SD", np.round(responses.std(0), 4),
          "Fisher cost", round(float(direction @ r267_Fisher @ direction), 6))

print("Finite-difference seconds:", round(time.time() - t268, 1))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 268: verified probability capture and MAP route fit\n"
            "- Corrected latent-dimension handling; verified probability-derived means before fitting.\n"
            "- Fitted minimum-categorical-distortion routes by finite differences and linear algebra only.\n"
            "- Submitted glucose-only weights unchanged.\n"
            "- Python runs: 268; forward passes: 24.\n")
```