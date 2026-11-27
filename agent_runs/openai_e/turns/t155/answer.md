The best attention variant still has hidden-state error of 5.77, so the reconstruction remains unreliable. I’ll bypass it by recovering the input to the final coda from the GPU hidden states. If replaying the coda reproduces those states closely, we can test immediate weight perturbations without relying on the faulty recurrent reconstruction.

```python
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)

gpu_h134 = torch.as_tensor(core_state[:, 192:960].reshape(-1, 4, 192),
                           dtype=torch.float32)
gpu_out134 = torch.as_tensor(core_state[:, :192], dtype=torch.float32)
check_out134 = core_norm(gpu_h134.mean(1), "core.out_norm")
print("GPU h -> out consistency RMSE:",
      float(torch.sqrt(torch.mean((check_out134 - gpu_out134) ** 2))))

def coda_input134(x):
    y = core_layer(x, "core.coda.0", 4)
    return y, core_norm(y.mean(1), "core.out_norm")

precoda134 = torch.empty_like(gpu_h134)
inverse_errors134 = []
for start134 in range(0, len(gpu_h134), 256):
    target134 = gpu_h134[start134:start134 + 256]
    x134 = target134.clone()
    for iteration134 in range(30):
        y134, _ = coda_input134(x134)
        error134 = target134 - y134
        if float(error134.abs().max()) < 2e-5:
            break
        x134 = x134 + error134
    precoda134[start134:start134 + len(target134)] = x134
    final134, _ = coda_input134(x134)
    inverse_errors134.append(float(torch.sqrt(torch.mean((final134-target134) ** 2))))

replay_h134 = torch.empty_like(gpu_h134)
replay_out134 = torch.empty_like(gpu_out134)
for start134 in range(0, len(gpu_h134), 256):
    yy134, oo134 = coda_input134(precoda134[start134:start134+256])
    replay_h134[start134:start134+len(yy134)] = yy134
    replay_out134[start134:start134+len(oo134)] = oo134
rmse_h134 = float(torch.sqrt(torch.mean((replay_h134-gpu_h134) ** 2)))
rmse_o134 = float(torch.sqrt(torch.mean((replay_out134-gpu_out134) ** 2)))
print("Coda replay hidden RMSE:", rmse_h134,
      "out RMSE:", rmse_o134,
      "worst chunk hidden RMSE:", max(inverse_errors134))
print("Coda-input scale:", float(precoda134.std()))
print("Original c_out singular-value range:",
      float(torch.linalg.svdvals(W0["core.c_out.weight"]).min()),
      float(torch.linalg.svdvals(W0["core.c_out.weight"]).max()))

assert np.isfinite(rmse_h134) and np.isfinite(rmse_o134)
with open("/work/notes.md", "a") as f134:
    f134.write("\n## Run 134: bypass uncertain recurrent reconstruction\n"
               "Run133 best raw-state RMSE remained 5.77. Recovered pre-coda inputs "
               "from original GPU h using forward-only fixed-point inversion; no "
               "gradients or weight changes. Replay h RMSE=%.8g, out RMSE=%.8g. "
               "This supplies an exact local anchor for immediate b_in perturbations, "
               "not a validated full rollout reconstruction. Python runs: 134; "
               "forwards: 15.\n" % (rmse_h134, rmse_o134))
```