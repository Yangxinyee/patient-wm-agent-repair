The labels match exactly. Two neurons have negligible estimated factual contributions and can host a norepinephrine correction. The gate is reliable after averaging latent draws, but not perfect on every draw, so I’ll test the actual GELU gating formula. The next scan will show whether any probe-calibrated correction fits the remaining NLL budget.

```python
from scipy.special import ndtr

torch.set_num_threads(8)
r138_slots = r137_neuron_rank[:2].astype(int)
r138_removed = -F.linear(
    r137_Hbase[..., r138_slots] - r137_hmean[r138_slots],
    r137_V[:, r138_slots]).numpy().astype(np.float64)
r138_L0 = r137_Lbase.numpy().astype(np.float64) + r138_removed
r138_y = r137_bins.numpy()
r138_original_loss = r137_original_nll.numpy()
r138_base_loss = r137_nllbase.numpy()

def r138_nll(logits):
    ex = np.exp(logits - logits.max(-1, keepdims=True))
    probs = ex / ex.sum(-1, keepdims=True)
    return -np.log(np.maximum(
        probs.mean(0)[np.arange(len(r138_y)), r138_y], 1e-35))

r138_p = r137_Pbase.numpy().astype(np.float64)
r138_mu = (r138_p * r137_centres[None]).sum(-1)
r138_var = ((r138_p * r137_centres[None]**2).sum(-1) - r138_mu**2).clip(0)
r138_active_fit = r132_family & r132_active & r137_fit
r138_active_val = r132_family & r132_active & r137_val
r138_tau = r138_var[:, r138_active_fit].mean()
r138_map_centres = r137_centres[np.flatnonzero(r132_family)[0]]
r138_tilt = (r138_map_centres - r138_mu[:, r138_active_fit].mean()) / r138_tau
r138_G = 2.0 * r137_G
r138_Gprobe = 2.0 * ((r137_Xnp + r131_dx) @ r132_gv + r132_gb)
r138_alpha = 0.20

def r138_pair(g, f):
    a = g + r138_alpha*f
    b = g - r138_alpha*f
    return (a*ndtr(a) - b*ndtr(b)) / (2*r138_alpha)

r138_loss0 = r138_nll(r138_L0)
print("reused neurons:", r138_slots.tolist(), "MAP variance scale:", r138_tau)
print("removal-only NLL increment:",
      (r138_loss0-r138_base_loss).mean())
print("columns: ridge, target_probe_shift, total_NLL_delta_all/fit/validation,"
      " incremental_MAP_delta, incremental_glucose_delta")

r138_features = {}
r138_results = []
for ridge138, (v138, b138) in r132_candidates.items():
    f138, df138 = r137_probe_features[ridge138]
    centre138 = f138[:, r138_active_fit].mean()
    pair138 = r138_pair(r138_G, f138-centre138)
    pairprobe138 = r138_pair(r138_Gprobe, f138+df138-centre138)
    derivative138 = ((pairprobe138-pair138) * r138_var/r138_tau)[
        :, r138_active_fit].mean()
    r138_features[ridge138] = {
        "v": v138, "b": b138-centre138, "pair": pair138,
        "probe_derivative": float(derivative138)
    }
    for delta138 in (0.10, 0.20, 0.329011):
        scale138 = delta138 / derivative138
        loss138 = r138_nll(
            r138_L0 + scale138*pair138[..., None]*r138_tilt)
        row138 = {
            "ridge": ridge138, "probe_shift": delta138, "scale": scale138,
            "all": float((loss138-r138_original_loss).mean()),
            "fit": float((loss138-r138_original_loss)[r137_fit].mean()),
            "validation": float((loss138-r138_original_loss)[r137_val].mean()),
            "increment_MAP": float((loss138-r138_base_loss)[r132_family].mean()),
            "increment_glucose": float((loss138-r138_base_loss)[~r132_family].mean())
        }
        r138_results.append(row138)
        print(ridge138, delta138,
              *(round(row138[k], 6) for k in
                ("all", "fit", "validation", "increment_MAP", "increment_glucose")))

with Path("/work/notes.md").open("a") as note138:
    note138.write("\n## Runs 137–138: verified local NE scan\n"
                  "Cached labels/probabilities verified exactly. Strength-2 insulin local NLL delta "
                  "is +0.002618 overall, +0.001343 on 75 fit stays and +0.006453 on 25 held-out stays. "
                  "Selected low-contribution neurons " + str(r138_slots.tolist()) + ". "
                  "Scanning covariance-constrained NE directions with the actual paired-GELU gate. "
                  "Probe shifts are diagnostics, not true dose-effect targets. No new edits installed.\n"
                  + repr(r138_results) + "\n")
```
