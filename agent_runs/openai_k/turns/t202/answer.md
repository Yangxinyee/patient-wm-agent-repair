The reserved channels encode doubled-dose contrasts close to one-for-one in physical rate units. That points to a routing problem, though their factual residuals are noisier than the identification residuals. I’ll measure the actual value-head response to small token perturbations and fit low-distortion directions. Modest, generalizing directions would support this route; very large ones would indicate saturation risk.

```python
import numpy as np, torch, time
from scipy.linalg import solve
torch.set_num_threads(8)

r186_start = time.time()
r186_sensors = list(r49_sensors)
r186_sel = np.concatenate([
    np.flatnonzero((r179_drug == d) & ~r180_case_test)[
        np.linspace(0, np.sum((r179_drug == d) & ~r180_case_test)-1, 32).astype(int)]
    for d in (0, 1)])
r186_n = len(r186_sel)
r186_idx = np.r_[r179_indices[r186_sel], r179_indices[r186_sel]]
r186_u = torch.cat([r179_u0[r186_sel], uc185[r186_sel]], dim=0)
r186_perturb = torch.zeros(192)
r186_dirs = torch.zeros(192, 2)
r186_oldlin = lin0
r186_backup = {k: W0[k].clone() for k in base185}

def r186_lin(x, prefix):
    y = r186_oldlin(x, prefix)
    if prefix == "z_in":
        y = y + r186_perturb + x[:, 30:32] @ r186_dirs.T
    return y

def r186_heads(indices, actions):
    out, h = r59_core(indices, actions)
    return r179_capture_head(out, h, indices)

def r186_means(heads):
    return np.column_stack([np.asarray(heads[s]["M"]).reshape(-1)
                            for s in r186_sensors])

try:
    for k, v in base185.items():
        W0[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32).clone()
    globals()["lin0"] = r186_lin
    r186_baseheads = r186_heads(r186_idx, r186_u)
    r186_M = r186_means(r186_baseheads)
    r186_J = np.empty((2*r186_n, 4, 192), dtype=np.float64)
    r186_eps = 0.05
    for j in range(192):
        r186_perturb.zero_()
        r186_perturb[j] = r186_eps
        plus = r186_means(r186_heads(r186_idx, r186_u))
        r186_perturb[j] = -r186_eps
        minus = r186_means(r186_heads(r186_idx, r186_u))
        r186_J[:, :, j] = (plus-minus)/(2*r186_eps)
        if (j+1) % 64 == 0:
            print("FD dimensions:", j+1, "elapsed seconds:", round(time.time()-r186_start, 1))
    r186_perturb.zero_()

    r186_z0 = np.asarray(z0185)[r186_sel, 30:32]
    r186_zc = np.asarray(zc185)[r186_sel, 30:32]
    r186_A = np.concatenate([
        r186_J[r186_n:] * r186_zc[:, d, None, None]
        - r186_J[:r186_n] * r186_z0[:, d, None, None]
        for d in (0,1)], axis=2)
    r186_B = np.concatenate([
        r186_J[:r186_n] * r186_z0[:, d, None, None]
        for d in (0,1)], axis=2)
    r186_var = []
    for s in r186_sensors:
        p = np.asarray(r186_baseheads[s]["P"])[:r186_n]
        if p.ndim == 3:
            p = p.mean(axis=1)
        c = np.asarray(r186_baseheads[s]["C"]).reshape(-1)
        mu = np.sum(p*c, axis=-1)
        r186_var.append(np.maximum(np.sum(p*c*c, axis=-1)-mu*mu, 1.0))
    r186_var = np.column_stack(r186_var)
    r186_scale = np.sqrt(r186_var)
    r186_target = np.column_stack([
        np.asarray(r180_effects[s])[r179_drug[r186_sel]] * r179_dc[r186_sel]
        for s in r186_sensors])
    r186_y = r186_target-(r186_M[r186_n:]-r186_M[:r186_n])
    A = (r186_A/r186_scale[:,:,None]).reshape(-1,384)
    B = (r186_B/r186_scale[:,:,None]).reshape(-1,384)
    y = (r186_y/r186_scale).reshape(-1)
    gram = A.T@A/len(y)
    fisher = B.T@B/len(y)
    rhs = A.T@y/len(y)
    r186_solutions = []
    print("sensor mean variances:", dict(zip(r186_sensors, r186_var.mean(axis=0).round(3))))
    print("RMS token Jacobians:", dict(zip(r186_sensors,
          np.sqrt(np.mean(r186_J*r186_J, axis=(0,2))).round(5))))
    print("ridge | direction norms | linear effect RMSE | approximate factual KL")
    for alpha in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        v = solve(gram + fisher + alpha*np.eye(384), rhs, assume_a="pos")
        dirs = np.column_stack([v[:192], v[192:]])
        err = float(np.sqrt(np.mean((A@v-y)**2)))
        kl = float(0.5*np.mean((B@v)**2))
        r186_solutions.append(dict(alpha=alpha, dirs=dirs, error=err, kl=kl))
        print(f"{alpha:.0e} | {np.linalg.norm(dirs,axis=0).round(3)} | {err:.5f} | {kl:.6f}")
finally:
    globals()["lin0"] = r186_oldlin
    for k,v in r186_backup.items():
        W0[k] = v
    r186_perturb.zero_()
    r186_dirs.zero_()

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 186: actual-head finite-difference routing\n"
            "- Reserved dose contrasts are approximately physical-rate contrasts (insulin 1.016, NE 1.108).\n"
            "- Measured exact local token-to-value-head finite differences and fit no-gradient directions with factual penalty.\n"
            "- Retained Jacobians and candidates; scratch weights restored. Saved edits unchanged.\n"
            "- Python runs: 186; forward passes: 10.\n")
```