A learned empty-action vector reduced the error substantially, but it did not produce an exact match. I’ll now check missing-unit handling, bin-edge conventions, and the remaining embedding terms. If those explain the mismatch, the posterior should match across bundles—not merely have a lower average error.

```python
print("=== EMBEDDING TABLE SIZES ===")
print({k: tuple(v.shape) for k, v in rm_W.items()
       if k.startswith("embed.") and (
           "categorical" in k or k in ("embed.kind.weight", "embed.ekind.weight",
                                      "embed.kq.weight"))})
print("numeric note:", VOCAB.get("numeric_note"))
print("config:", VOCAB["config"])

r109_statefile = np.load("/work/fwd_4/states_original.npz")
print("\nSaved counts versus CSV counts, first bundles:")
for j in range(6):
    idx = int(r103_idx[j])
    b = r103_bundle_rows[j]
    print(idx, r109_statefile["hour"][idx],
          int(r109_statefile["n_atoms"][idx]), len(b))
r109_ctx_check = (
    r103_prev[:, :192]
    + rm_ln(r103_hflow.mean(1), "core.out_norm")
    + rm_mlp(torch.log(r103_dt.clamp_min(1e-6)) / 10, "gap")
)
print("Context recheck RMSE:",
      float((r103_ctx - r109_ctx_check).square().mean().sqrt()))

r109_no_kq_base = (
    rm_W["embed.code.weight"][r103_codes]
    + rm_W["embed.kind.weight"][0]
    + rm_lin(r103_numeric, "embed.numeric")
)
r109_results = []
r109_best = None
r109_units = [-1] + list(range(rm_W["embed.categorical.unit.weight"].shape[0]))
r109_kqs = [-1] + list(range(rm_W["embed.kq.weight"].shape[0]))
r109_zero = torch.zeros(192)

for side in ("left", "right"):
    bins = []
    for row in r103_flat.itertuples(index=False):
        spec = rm_normspec[(row.event, "value")]
        bins.append(1 + np.searchsorted(spec["fine_bin_edges"], row.value, side=side))
    base = r109_no_kq_base + rm_W["embed.bins.value.weight"][
        torch.tensor(bins, dtype=torch.long)]
    for ek in range(rm_W["embed.ekind.weight"].shape[0]):
        for kq in r109_kqs:
            shared = base + rm_W["embed.ekind.weight"][ek]
            if kq >= 0:
                shared = shared + rm_W["embed.kq.weight"][kq]
            for gu in r109_units:
                for mu in r109_units:
                    gv = r109_zero if gu < 0 else rm_W["embed.categorical.unit.weight"][gu]
                    mv = r109_zero if mu < 0 else rm_W["embed.categorical.unit.weight"][mu]
                    unit_vectors = torch.where(
                        torch.as_tensor(r103_is_glucose, dtype=torch.bool).reshape(-1, 1),
                        gv, mv)
                    atoms = shared + unit_vectors
                    encoded = rm_mlp(atoms, "encode.mlp1")
                    pooled = torch.zeros(r108_B, 192)
                    pooled.index_add_(0, r108_bundle_index, encoded)
                    e = rm_ln(rm_mlp(pooled / r108_count.sqrt() + r108_size,
                                    "encode.mlp2"), "encode.norm")
                    for action_name, u in r108_actions.items():
                        z = rm_mlp(torch.cat([r103_ctx, e, u], 1), "z_post")[:, :32]
                        error = float((z - r103_target).square().mean().sqrt())
                        spec = (side, ek, kq, gu, mu, action_name)
                        r109_results.append((error, spec))
                        if r109_best is None or error < r109_best["rmse"]:
                            r109_best = {"rmse": error, "spec": spec,
                                         "e": e.clone(), "u": u.clone(), "z": z.clone()}
print("\nBest extended candidates:")
for item in sorted(r109_results)[:8]:
    print(item)
```
