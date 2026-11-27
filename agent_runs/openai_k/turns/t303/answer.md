The stronger glucose edit passes this sample: factual loss increased by 0.00383 nats. That leaves little room for MAP corrections. I’ll use finite differences—no gradients—to find a MAP direction that minimally disturbs the predicted value distributions. Lower categorical distortion than the rejected route would support testing it.

```python
import time

torch.set_grad_enabled(False)
torch.set_num_threads(4)

print("Tuning stays:", len(ids266), "next batch endpoints:",
      ids266[100], ids266[199])
print("Sensor order:", r49_sensors)
print("Centre-table metadata:", compact265(r49_centres))

def r267_full_pre(pre, ids):
    h = audit_block(linear228(pre, "core.c_out"), "core.coda.0")
    out = norm0(h.mean(1), "core.out_norm")
    captured = []
    old_softmax = torch.softmax

    def capture(x, dim, *args, **kwargs):
        p = old_softmax(x, dim, *args, **kwargs)
        if p.ndim == 2 and p.shape[-1] == 17:
            captured.append(np242(p).copy())
        return p

    torch.softmax = capture
    try:
        means = r49_readouts(out, h, ids)
    finally:
        torch.softmax = old_softmax
    return np242(means), captured

r267_sel = np.asarray(r244_sel, dtype=int)
r267_ids = np.asarray(r244_ids, dtype=int)
r267_case = r245_cases[("factual", 1.0)]
r267_pre_np = np242(r267_case["pre"])[r267_sel].copy()
r267_pre_np += (
    r266_I
    * np.einsum("nsj,ij->nsi",
                np242(r267_case["E"])[r267_sel], np.asarray(r255_CI))
    * np.asarray(r245_delta)[None, None, :]
)
r267_pre = torch.as_tensor(r267_pre_np, dtype=torch.float32)
r267_means, r267_captured = r267_full_pre(r267_pre, r267_ids)
print("Captured probability shapes:", [p.shape for p in r267_captured])

r267_ok = (
    len(r267_captured) == 4
    and all(p.shape == (len(r267_sel), 17) for p in r267_captured)
)
if r267_ok:
    r267_P = np.stack(r267_captured, axis=1).astype(np.float64)
    if isinstance(r49_centres, dict):
        r267_centres = np.stack([np.asarray(r49_centres[s]) for s in r49_sensors])
    else:
        r267_centres = np.asarray(r49_centres)
    assert r267_centres.shape == (4, 17), r267_centres.shape

    t267 = time.time()
    r267_eps = 0.025
    r267_Jlog = np.empty((len(r267_sel), 4, 17, 192), dtype=np.float64)
    for j in range(192):
        perturb = torch.zeros((1, 1, 192), dtype=torch.float32)
        perturb[..., j] = r267_eps
        _, pp = r267_full_pre(r267_pre + perturb, r267_ids)
        _, pm = r267_full_pre(r267_pre - perturb, r267_ids)
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

    print("Existing MAP direction mean response:",
          r267_A @ np.asarray(rr_Ndirection))
    print("Existing direction Fisher cost:",
          float(np.asarray(rr_Ndirection) @ r267_Fisher @ np.asarray(rr_Ndirection)))

    for ridge in [0.001, 0.01, 0.1, 1.0]:
        metric = r267_Fisher + ridge * r267_scale * np.eye(192)
        U = np.linalg.solve(metric, r267_A.T)
        direction = U @ np.linalg.pinv(r267_A @ U, rcond=1e-9) @ r267_desired
        r267_routes[ridge] = direction
        responses = np.einsum("nci,i->nc", r267_Jmean, direction)
        print("ridge", ridge, "norm", np.linalg.norm(direction),
              "mean", np.round(responses.mean(0), 4),
              "response SD", np.round(responses.std(0), 4),
              "Fisher cost", float(direction @ r267_Fisher @ direction))
    print("Finite-difference seconds:", round(time.time() - t267, 1))
else:
    print("Probability layout needs inspection; no derivative calculation attempted.")
    print("Head globals:", r49_readouts.__code__.co_names)
    print("Capture helper return variables:", r179_capture_head.__code__.co_varnames)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 267: full-distribution MAP routing\n"
            "- Forward 24 glucose strength 1.15 factual delta +0.00383 passes on the original 100-stay batch.\n"
            "- Computing finite-difference value-probability Jacobians and minimum-Fisher-cost MAP routes.\n"
            "- No autograd, backpropagation or weight optimiser used; submitted weights unchanged.\n"
            "- Python runs: 267; forward passes: 24.\n")
```