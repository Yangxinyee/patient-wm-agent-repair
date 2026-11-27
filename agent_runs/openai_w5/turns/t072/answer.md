The best decoder uses context and action information, but we still need a way to turn its output into the correct physiological change. I’ll measure the next-reading response to small latent perturbations, using finite differences rather than gradients. I’ll also check whether the fuller measurement encoding improves reconstruction. Accurate recurrence and consistent drug-specific responses would justify a low-rank posterior edit; large reconstruction errors or unstable responses would not.

```python
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

# Check the more complete bundle encoding before using posterior features.
ctx61, hf61, dt61 = context_from_saved(rows55, s4)
PH61 = F.gelu(wm_lin(torch.cat(
    [ctx61, wm_e4[rows55], U4[rows55]], dim=1), "z_post.0"))
z61 = wm_lin(PH61, "z_post.2")[:, :32].numpy()
zerr61 = np.sqrt(np.mean(
    (z61 - s4["x"][rows55, 960:992])**2, axis=1))
print("Complete-encoding latent RMSE:", float(np.sqrt(np.mean(zerr61**2))))
for active in [False, True]:
    ee = zerr61[present55.any(1) == active]
    print("Action bundle", active, "median/max RMSE",
          float(np.median(ee)), float(ee.max()))

def latent_step61(j, z_batch):
    """Original recurrent step, varying only its supplied posterior mean."""
    z_batch = torch.as_tensor(z_batch, dtype=torch.float32)
    K = len(z_batch)
    hh = torch.from_numpy(s4["x"][j-1:j, 192:960]).reshape(1, 4, 192)
    dt = float((s4["hour"][j] - s4["hour"][j-1]) * 3600)
    hh = hh * torch.exp(-dt * torch.exp(W0["core.flow.a"]))
    hh = hh.expand(K, -1, -1)
    ee = wm_e4[j:j+1].expand(K, -1)
    uu = U4[j:j+1].expand(K, -1)
    tokens = torch.cat([
        hh + W0["core.slots"], ee[:, None], uu[:, None],
        wm_lin(z_batch, "z_in")[:, None]], dim=1)
    E = wm_norm(wm_layer(tokens, "core.prelude.0")[:, :4],
                "core.prelude_norm")
    start = int(np.searchsorted(s4["stay_id"], s4["stay_id"][j]))
    begin = max(start, j - 128)
    memory = wm_e4[begin:j][None].expand(K, -1, -1)
    ages = torch.as_tensor(
        (s4["hour"][j] - s4["hour"][begin:j]) * 3600,
        dtype=torch.float32).reshape(1, -1, 1)
    age_vectors = wm_mlp(torch.log1p(ages), "core.age").expand(K, -1, -1)
    bb = hh + E
    for layer in range(2):
        bb = wm_layer(bb, f"core.blocks.0.{layer}",
                      memory, age_vectors, "input")
    delta = F.softplus(W0["core.delta_in"])
    abar = torch.exp(-delta * torch.exp(W0["core.a_in"]))
    new_h = abar * hh + delta * wm_lin(E, "core.b_in") + bb - (hh + E)
    return wm_layer(wm_lin(new_h, "core.c_out"), "core.coda.0")

def next_means61(j, h_branches):
    rr = np.flatnonzero(np.asarray(cacheTrain["previous"]) == j)
    if not len(rr):
        return None, None
    K, R = len(h_branches), len(rr)
    ix = np.tile(rr, K)
    cc = {}
    n = len(cacheTrain["df"])
    for key, value in cacheTrain.items():
        if isinstance(value, pd.DataFrame):
            cc[key] = value.iloc[ix].reset_index(drop=True)
        elif isinstance(value, (np.ndarray, torch.Tensor)) and len(value) == n:
            cc[key] = value[ix]
        else:
            cc[key] = value
    hh = h_branches[:, None].expand(K, R, 4, 192).reshape(K*R, 4, 192)
    dt = torch.as_tensor(cc["dt"], dtype=torch.float32)
    flowed = hh * torch.exp(-dt[:, None] * torch.exp(W0["core.flow.a"]))
    cc["base"] = (wm_norm(hh.mean(1), "core.out_norm") +
                  wm_norm(flowed.mean(1), "core.out_norm"))
    result = head_features_layout(
        cc, head_layout_spec["layout"], head_layout_spec["ageunit"],
        nsamples=1, seed=913)
    means = as_np(result["means"]).reshape(K, R)
    events = cacheTrain["df"].iloc[rr]["event"].to_numpy()
    return means, events

available61 = np.intersect1d(
    rows55[valid55 & present55.any(1)],
    np.unique(np.asarray(cacheTrain["previous"])))
sample61 = available61[np.linspace(0, len(available61)-1, 16).astype(int)]
eps61 = 0.05
eye61 = np.eye(32, dtype=np.float32)
jac61, recurrence_errors61 = [], []

# Deterministic prior means eliminate Monte Carlo noise in local differences.
old_pw61 = W0["z_prior.2.weight"]
old_pb61 = W0["z_prior.2.bias"]
W0["z_prior.2.weight"] = old_pw61.clone()
W0["z_prior.2.bias"] = old_pb61.clone()
W0["z_prior.2.weight"][32:] = 0
W0["z_prior.2.bias"][32:] = -30
try:
    for j in sample61:
        zz = s4["x"][j, 960:992]
        zb = np.vstack([zz, zz + eps61*eye61, zz - eps61*eye61])
        hb = latent_step61(int(j), zb)
        recurrence_errors61.append(float(torch.sqrt(torch.mean(
            (hb[0] - torch.from_numpy(
                s4["x"][j, 192:960]).reshape(4, 192))**2))))
        means, events = next_means61(int(j), hb)
        JJ = []
        for prefix in ["glucose", "map"]:
            mask = np.char.startswith(events.astype(str), prefix)
            if not mask.any():
                JJ.append(np.full(32, np.nan))
            else:
                JJ.append(((means[1:33, mask] - means[33:65, mask]) /
                           (2*eps61)).mean(1))
        jac61.append(JJ)
finally:
    W0["z_prior.2.weight"] = old_pw61
    W0["z_prior.2.bias"] = old_pb61

jac61 = np.asarray(jac61)
J61 = np.nanmean(jac61, axis=0)
ridge61 = 0.001 * np.trace(J61 @ J61.T) / 2
target61 = np.diag([
    dyn_estimates["G"]["beta"] * scale55[0],
    dyn_estimates["M"]["beta"] * scale55[1]])
V61 = J61.T @ np.linalg.solve(
    J61 @ J61.T + ridge61*np.eye(2), target61)

print("Recurrence RMSE median/max:",
      float(np.median(recurrence_errors61)), float(max(recurrence_errors61)))
print("Mean response Jacobian singular values:", np.linalg.svd(J61, compute_uv=False))
print("Latent correction column norms:", np.linalg.norm(V61, axis=0))
print("Mean next-reading shifts for unit normalized innovations:\n", J61 @ V61)
response61 = np.einsum("nij,jk->nik", jac61, V61)
print("Across-state response median:\n", np.nanmedian(response61, axis=0))
print("Across-state response SD:\n", np.nanstd(response61, axis=0))
print("Edits:", edited())
```
