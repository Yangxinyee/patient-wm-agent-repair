The retry reached the reconstructed core but failed on a tensor-shape mismatch, likely from reversing the helper’s output and slot-state return order. No sensitivity result was produced. I’ll identify the outputs by rank and rerun; this fixes the test harness rather than changing the model.

```python
def r51_core(z):
    a51, b51 = repaired_core_local(
        core_probe_h, core_probe_e, core_probe_u, z,
        core_probe_memory, core_probe_mask, core_probe_ages)
    out51, h51 = (a51, b51) if a51.ndim == 2 else (b51, a51)
    assert out51.shape == (64, 192) and h51.shape == (64, 4, 192)
    return out51, h51

r51_t0 = time.time()
r49_out0, r49_h0 = r51_core(core_probe_z)
print("Corrected shapes:", r49_out0.shape, r49_h0.shape,
      "flow", W0["core.flow.a"].shape)
r49_mean0 = r49_readouts(r49_out0, r49_h0, core_probe_indices)
r49_eps = 0.02
r49_J = np.zeros((64, 4, 32))
for j51 in range(32):
    zp51, zm51 = core_probe_z.clone(), core_probe_z.clone()
    zp51[:, j51] += r49_eps
    zm51[:, j51] -= r49_eps
    op51, hp51 = r51_core(zp51)
    om51, hm51 = r51_core(zm51)
    r49_J[:, :, j51] = (
        (r49_readouts(op51, hp51, core_probe_indices)
         - r49_readouts(om51, hm51, core_probe_indices))/(2*r49_eps)).numpy()

r49_targets = np.array([[-1.7, -1.7, 0, 0], [0, 0, 2.8, 2.8]]).T
r49_A = (r49_J/r49_scales[None, :, None]).reshape(-1, 32)
r49_B = np.tile(r49_targets/r49_scales[:, None], (64, 1))
r49_directions = {}
print("Baseline sensor means:", r49_mean0.mean(0).numpy())
print("Mean sensitivity singular values:",
      np.linalg.svd(r49_J.mean(0), compute_uv=False))
for alpha51 in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
    K51 = np.linalg.solve(
        r49_A.T @ r49_A + alpha51*np.eye(32), r49_A.T @ r49_B)
    r49_directions[alpha51] = K51
    response51 = np.einsum("nsz,zd->nsd", r49_J, K51)
    print("ridge", alpha51, "latent norms", np.linalg.norm(K51, axis=0),
          "mean response", response51.mean(0).round(3).tolist(),
          "response SD", response51.std(0).round(3).tolist())
print("Elapsed seconds:", round(time.time()-r51_t0, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 49–51: sensitivity harness repairs\n"
            "- Closed archive in run 49 and reversed core return order in run 50 produced no sensitivity evidence.\n"
            "- Run 51 identifies core outputs by tensor rank and computes finite-difference sensor sensitivities.\n"
            "- No tensors edited. Python runs: 51; forward passes: 4.\n")
```