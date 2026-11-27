The synthetic inputs used an outdated last-value register layout. Its offset cancels in feature differences, but not in predicted means or sensitivities. I’ll correct that layout and cache paired CPU logits for fitting. If the corrected factual calculation disagrees materially with the GPU baseline, I’ll stop and audit it rather than trust the edit.

```python
import numpy as np, torch
torch.set_num_threads(8)
sc_names = ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
            "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]
sc_original = {n: to_np0(load_original(n)).copy() for n in sc_names}
sc_checkpoint = {n: to_np0(load(n)).copy() for n in sc_names}
sc_nodes = np.asarray(graft_nodes, dtype=int)
sc_rank = to_np0(rank_template).astype(np.float32)
sc_gate_w = sc_checkpoint[sc_names[0]][sc_nodes[1]].copy()
sc_gate_b = float(sc_checkpoint[sc_names[1]][sc_nodes[1]])
sc_bias_comp = sc_checkpoint[sc_names[3]] - sc_original[sc_names[3]]
sc_meta = {"first": head_meta_train, "extra": head_meta_extra, "tuning": head_meta_tune}
sc_sf = {"first": state_files[4], "extra": state_files[5], "tuning": state_files[3]}

def sc_cache_inputs(x, trans, meta):
    x = np.asarray(x, dtype=np.float32)
    xt = torch.from_numpy(x)
    h = torch.nn.functional.gelu(
        xt @ torch.from_numpy(sc_original[sc_names[0]]).T
        + torch.from_numpy(sc_original[sc_names[1]]))
    original_logits = (h @ torch.from_numpy(sc_original[sc_names[2]]).T
                       + torch.from_numpy(sc_original[sc_names[3]])
                       + torch.as_tensor(trans, dtype=torch.float32)[None])
    removed = h[..., sc_nodes] @ torch.from_numpy(
        sc_original[sc_names[2]][:, sc_nodes]).T
    base_logits = original_logits - removed + torch.from_numpy(sc_bias_comp)
    centres = np.stack([probe_centres_by_event[e] for e in meta.event]).astype(np.float32)
    bins = meta.observed_bin.to_numpy(dtype=int)
    original_p = torch.softmax(original_logits, -1).mean(0).numpy()
    base_p = torch.softmax(base_logits, -1).mean(0).numpy()
    gate = x @ sc_gate_w + sc_gate_b
    return {"x": x, "logits": base_logits.numpy(),
            "gate": gate, "centres": centres, "bins": bins,
            "original_nll": float(-np.log(np.maximum(
                original_p[np.arange(len(meta)), bins], 1e-30)).mean()),
            "base_mean": (base_p * centres).sum(-1),
            "original_mean": (original_p * centres).sum(-1),
            "meta": meta}

sc_fact = {}
for tag0 in ["first", "extra", "tuning"]:
    trans0 = to_np0(cpu_head_parts(sc_meta[tag0], sc_sf[tag0])["transition"])
    sc_fact[tag0] = sc_cache_inputs(robust_X[tag0], trans0, sc_meta[tag0])

sc_reg_offset = (lab_meta.last_z.to_numpy(np.float32)[:, None]
                 * to_np0(W["last_numeric.weight"][:, 6]
                          - W["last_numeric.weight"][:, 2])[None])
sc_syn = {}
for mult0 in [1.0, 0.5, 2.0]:
    x0 = lab_head_results[mult0]["features"] + sc_reg_offset[None]
    sc_syn[mult0] = sc_cache_inputs(x0, to_np0(parts_probe0["transition"]), lab_meta)

sc_eigenvalues, sc_eigenvectors = np.linalg.eigh(robust_projectors["register_prior"])
sc_Q = sc_eigenvectors[:, sc_eigenvalues > 0.5]
sc_active = {}
for tag0 in ["first", "extra"]:
    mask0 = (sc_meta[tag0].event.eq("glucose_finger")
             & sc_meta[tag0].has_last.gt(0)).to_numpy()
    sc_active[tag0] = sc_fact[tag0]["x"][:, mask0].reshape(-1, 192).astype(np.float64)
sc_F = np.concatenate(list(sc_active.values()))
sc_mean = sc_F.mean(0)
sc_Fq = (sc_F - sc_mean) @ sc_Q
sc_cov = sc_Fq.T @ sc_Fq / len(sc_Fq)

sc_dx, sc_y, sc_mult, sc_stay = [], [], [], []
for mult0 in [0.5, 2.0]:
    b0 = sc_syn[1.0]
    bm = sc_syn[mult0]
    p0 = torch.softmax(torch.from_numpy(b0["logits"]), -1).numpy()
    rm0 = (p0 * sc_rank).sum(-1, keepdims=True)
    sens0 = (p0 * (sc_rank - rm0) * b0["centres"][None]).sum(-1).mean(0)
    dx0 = (bm["x"] - b0["x"]).mean(0).astype(np.float64)
    true_local0 = -1.92870928 * (mult0 - 1.0) * lab_rates
    base_local0 = bm["base_mean"] - b0["base_mean"]
    sc_dx.append(dx0 @ sc_Q)
    sc_y.append((true_local0 - base_local0) / np.maximum(sens0, 1e-3))
    sc_mult.extend([mult0] * len(lab_meta))
    sc_stay.extend(lab_meta.stay_id.tolist())
sc_D = np.concatenate(sc_dx)
sc_y = np.concatenate(sc_y).astype(np.float64)
sc_mult = np.asarray(sc_mult)
sc_stay = np.asarray(sc_stay)

print("Original CPU NLL", {k: v["original_nll"] for k, v in sc_fact.items()})
print("GPU tuning NLL", float(np.mean(head_gpu_nll)))
print("Protected rank", sc_Q.shape[1], "factual rows", len(sc_F))
print("Synthetic mean targets",
      {m: float(np.mean(-1.92870928 * (m-1) * lab_rates)) for m in [0.5, 2.0]})
print("Original corrected local means",
      {m: float(np.mean(sc_syn[m]["original_mean"] - sc_syn[1.0]["original_mean"]))
       for m in [0.5, 2.0]})
print("Contrast RMS raw/projected", {
    m: (float(np.sqrt(np.mean((sc_syn[m]["x"]-sc_syn[1.0]["x"])**2))),
        float(np.sqrt(np.mean(sc_D[sc_mult == m]**2))))
    for m in [0.5, 2.0]})
print("Glucose gate quantiles", np.quantile(
    sc_fact["tuning"]["gate"][:, head_meta_tune.event.eq("glucose_finger")], [.01,.5,.99]))
print("MAP gate quantiles", np.quantile(
    sc_fact["tuning"]["gate"][:, head_meta_tune.event.str.startswith("map")], [.01,.5,.99]))
```
