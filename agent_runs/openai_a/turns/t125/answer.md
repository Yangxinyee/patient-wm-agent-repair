The extended attribute search still did not validate the reconstruction. I’ll check state alignment and the time-flow context next, rather than assuming the best-looking attribute combination is correct. A source-row mismatch or a context variant that matches exactly would explain the failure; neither would justify a weight change by itself.

```python
r110_x = r109_statefile["x"]
print("Source target RMSE:",
      np.sqrt(np.mean((r103_target.numpy() - r110_x[r103_idx, 960:])**2)))
print("Source previous-state RMSE:",
      np.sqrt(np.mean((r103_prev.numpy() - r110_x[r103_idx - 1])**2)))
for j in range(5):
    idx = int(r103_idx[j])
    print("dt", float(r103_dt[j, 0]),
          float((r109_statefile["hour"][idx] -
                 r109_statefile["hour"][idx - 1]) * 3600))
print("Embedding zero-row norms:",
      {k: round(float(v[0].norm()), 5) for k, v in rm_W.items()
       if k.startswith("embed.") and v.ndim == 2 and (
           "bins." in k or "categorical." in k or
           k in ("embed.kind.weight", "embed.ekind.weight", "embed.kq.weight"))})

r110_contexts = {}
for flow_name, h in (
    ("flowed", r103_hflow),
    ("unflowed", r103_prev[:, 192:960].reshape(-1, 4, 192)),
):
    for gap_name, gt in (
        ("log_div10", torch.log(r103_dt.clamp_min(1e-6)) / 10),
        ("log1p_div10", torch.log1p(r103_dt) / 10),
        ("log_raw", torch.log(r103_dt.clamp_min(1e-6))),
        ("hours_div10", torch.log((r103_dt / 3600).clamp_min(1e-6)) / 10),
    ):
        r110_contexts[(flow_name, gap_name)] = (
            r103_prev[:, :192] + rm_ln(h.mean(1), "core.out_norm")
            + rm_mlp(gt, "gap"))

r110_results = []
r110_best = None
# Test both the documented attributes and the earlier best candidate.
for attr_spec in ((1, 0, 1, 2), (0, 0, 0, 2), (1, 1, 1, 2)):
    ek, kq, gu, mu = attr_spec
    unit_vectors = torch.where(
        torch.as_tensor(r103_is_glucose, dtype=torch.bool).reshape(-1, 1),
        rm_W["embed.categorical.unit.weight"][gu],
        rm_W["embed.categorical.unit.weight"][mu])
    atoms = (r109_no_kq_base
             + rm_W["embed.bins.value.weight"][torch.tensor(
                 [1 + np.searchsorted(rm_normspec[(r.event, "value")]["fine_bin_edges"],
                                      r.value, side="left")
                  for r in r103_flat.itertuples(index=False)], dtype=torch.long)]
             + rm_W["embed.ekind.weight"][ek]
             + rm_W["embed.kq.weight"][kq] + unit_vectors)
    encoded = rm_mlp(atoms, "encode.mlp1")
    pooled = torch.zeros(r108_B, 192)
    pooled.index_add_(0, r108_bundle_index, encoded)
    for pool_power in (0.0, 0.5, 1.0):
        e = rm_ln(rm_mlp(pooled / r108_count.pow(pool_power) + r108_size,
                        "encode.mlp2"), "encode.norm")
        for ctx_spec, ctx in r110_contexts.items():
            for action_name, u in r108_actions.items():
                z = rm_mlp(torch.cat([ctx, e, u], 1), "z_post")[:, :32]
                error = float((z - r103_target).square().mean().sqrt())
                spec = (attr_spec, pool_power, ctx_spec, action_name)
                r110_results.append((error, spec))
                if r110_best is None or error < r110_best["rmse"]:
                    r110_best = {"rmse": error, "spec": spec,
                                 "e": e.clone(), "u": u.clone(), "z": z.clone()}
print("Best context/pooling candidates:")
for item in sorted(r110_results)[:8]:
    print(item)
```
