The cached head is now verified: it uses GELU, and reconstructed probabilities agree within \(3\times10^{-7}\). I’ll measure candidate corrections against the actual reading labels and identify low-impact neurons to reuse. A promising direction should improve treatment sensitivity at a small held-out factual-loss cost.

```python
def r137_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("label/centre shapes:", np.shape(bin104), np.shape(C104))
r137_bins = torch.as_tensor(r137_np(bin104), dtype=torch.long).reshape(-1)
r137_centres = r137_np(C104).astype(np.float64)
assert len(r137_bins) == len(meta96)
assert r137_centres.shape == (len(meta96), 17)

r137_original_nll = -r136_P.mean(0).gather(1, r137_bins[:, None]).squeeze(1).log()
print("label check against cached NLL:",
      (r137_original_nll - cache98["nll"]).abs().max().item())

r137_W = {n: load(n).clone() for n in edited()}
r137_Hbase = F.gelu(F.linear(
    r136_X, r137_W["attr_heads.value_bin.0.weight"],
    r137_W["attr_heads.value_bin.0.bias"]))
r137_Lbase = F.linear(
    r137_Hbase, r137_W["attr_heads.value_bin.2.weight"],
    r137_W["attr_heads.value_bin.2.bias"]) + r136_transition
r137_Pbase = r137_Lbase.softmax(-1)
r137_nllbase = -r137_Pbase.mean(0).gather(1, r137_bins[:, None]).squeeze(1).log()

r137_fit_ids = np.unique(meta96.loc[r132_fit, "stay_id"])
r137_fit = meta96.stay_id.isin(r137_fit_ids).to_numpy()
r137_val = ~r137_fit
print("fit/validation stays:", len(r137_fit_ids), meta96.loc[r137_val, "stay_id"].nunique())
for label137, mask137 in (("all", np.ones(len(meta96), bool)),
                         ("fit", r137_fit), ("validation", r137_val)):
    print("strength-2 baseline", label137,
          "original", r137_original_nll[mask137].mean().item(),
          "edited", r137_nllbase[mask137].mean().item(),
          "delta", (r137_nllbase-r137_original_nll)[mask137].mean().item())

r137_Xnp = r136_X.numpy().astype(np.float64)
r137_G = r137_Xnp @ r132_gv + r132_gb
print("gate active/inactive minimum margins:",
      r137_G[:, r132_family & r132_active].min(),
      -r137_G[:, ~(r132_family & r132_active)].max())

r137_probe_features = {}
for ridge137, (v137, b137) in r132_candidates.items():
    feature137 = r137_Xnp @ v137 + b137
    response137 = r131_dx.astype(np.float64) @ v137
    active137 = r132_family & r132_active
    r137_probe_features[ridge137] = (feature137, response137)
    print("direction", ridge137,
          "intercept", b137,
          "active factual mean/SD",
          feature137[:, active137].mean(), feature137[:, active137].std(),
          "active probe response", response137[:, active137].mean(),
          "validation active probe response",
          response137[:, active137 & r137_val].mean())

r137_hmean = r137_Hbase[:, r137_fit].mean((0, 1))
r137_V = r137_W["attr_heads.value_bin.2.weight"]
r137_varV = (F.linear(r137_Pbase, r137_V.T.square())
             - F.linear(r137_Pbase, r137_V.T).square()).clamp_min(0)
r137_neuron_cost = (0.5 * (r137_Hbase-r137_hmean).square()
                   * r137_varV)[:, r137_fit].mean((0, 1)).numpy()
r137_neuron_cost[[107, 109]] = np.inf
r137_neuron_rank = np.argsort(r137_neuron_cost)
print("lowest estimated removal costs:",
      [(int(j), float(r137_neuron_cost[j])) for j in r137_neuron_rank[:12]])
print("MAP centres:", r137_centres[np.flatnonzero(r132_family)[0]].tolist())
```
