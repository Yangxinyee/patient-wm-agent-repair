The record-derived impulses persist for several hours, and the model has a 32-dimensional posterior pathway into its recurrent state. I’ll test whether small changes along that pathway produce the required immediate responses across many states. Stable, drug-specific responses would support a targeted edit; weak or highly variable responses would argue against it.

```python
import time
t56 = time.time()
o56, h56 = r51_core(core_probe_z)
idx56 = np.asarray(core_probe_indices, dtype=int)
base56 = r49_readouts(o56, h56, idx56)
print("Baseline core agreement:",
      float((o56-r49_out0).abs().max()),
      float((h56-r49_h0).abs().max()))
print("Aligned readout means:", base56.mean(0).tolist())
print("Sensor scales:", r49_scales)

eps56 = 0.05
J56 = np.empty((len(idx56), 4, 32), dtype=np.float64)
for j56 in range(32):
    zp56 = core_probe_z.clone()
    zm56 = core_probe_z.clone()
    zp56[:, j56] += eps56
    zm56[:, j56] -= eps56
    op56, hp56 = r51_core(zp56)
    om56, hm56 = r51_core(zm56)
    J56[:, :, j56] = (
        (r49_readouts(op56, hp56, idx56) -
         r49_readouts(om56, hm56, idx56)) / (2*eps56)).numpy()

target56 = np.array([[-1.93474217, 0.0],
                     [-2.01483673, 0.0],
                     [0.0, 3.05259229],
                     [0.0, 2.52816442]])
scale56 = np.array([2., 2., 3., 3.])
A56 = (J56 / scale56[None, :, None]).reshape(-1, 32)
B56 = np.broadcast_to(target56 / scale56[:, None],
                     (len(idx56), 4, 2)).reshape(-1, 2)
D56_candidates = {}
for lam56 in (0.0001, 0.001, 0.01, 0.1, 1.0):
    D56 = np.linalg.solve(A56.T@A56 + len(idx56)*lam56*np.eye(32),
                         A56.T@B56)
    D56_candidates[lam56] = D56
    resp56 = np.einsum("nsz,zd->nsd", J56, D56)
    print("ridge", lam56, "direction norms", np.linalg.norm(D56, axis=0),
          "mean responses", np.round(resp56.mean(0), 3).tolist(),
          "response SD", np.round(resp56.std(0), 3).tolist())

print("Action decoder:", rr39_action_decoder.coef_.shape,
      rr39_action_decoder.intercept_)
print("Centre fits:", {k: (v.coef_.shape, np.asarray(v.intercept_).tolist())
                       for k, v in rr41_center_models.items()})
print("Expected/innovation identity RMSE:",
      np.sqrt(np.mean((rr41_D[:rr41_n] -
                       rr41_expected - rr41_innov)**2, axis=0)))
print("Augmented inputs/hidden:", tuple(rr41_aug_inputs.shape), rr41_H.shape)
print("Seconds:", round(time.time()-t56, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 56: posterior response assay\n"
            "- Corrected readout bin dimensions and aligned probe registers.\n"
            "- Measured finite-difference responses of all 32 posterior coordinates "
            "and fitted two small, regularized drug-response directions.\n"
            "- No weights edited. Python runs: 56; forward passes: 4.\n")
```
