Ordinary measurement-only bundles also fail, so admissions are not the main cause. One untested possibility is that “no action” uses a learned empty-set vector rather than zero, as the empty world set does. I’ll test that alongside the measurement attributes; an exact posterior match would confirm the encoding.

```python
torch.set_num_threads(8)

def empty_set_vector(prefix, batch):
    pooled = torch.zeros(batch, 192)
    size = rm_lin(torch.zeros(batch, 1), prefix + ".size")
    return rm_ln(rm_mlp(pooled + size, prefix + ".mlp2"), prefix + ".norm")

r108_expected_base = (
    rm_W["embed.code.weight"][r103_codes]
    + rm_W["embed.kind.weight"][0]
    + rm_W["embed.kq.weight"][0]
    + rm_lin(r103_numeric, "embed.numeric")
    + rm_W["embed.bins.value.weight"][torch.tensor(r103_bins, dtype=torch.long)]
)
print("Base reconstruction RMSE:",
      float((r103_base - r108_expected_base).square().mean().sqrt()))
print("First numeric rows:", r103_numeric[:3].numpy())

r108_B = len(r103_counts)
r108_bundle_index = torch.repeat_interleave(
    torch.arange(r108_B), torch.as_tensor(r103_counts, dtype=torch.long))
r108_count = torch.as_tensor(r103_counts, dtype=torch.float32).reshape(-1, 1)
r108_size = rm_lin(torch.log1p(r108_count), "encode.size")
r108_actions = {
    "zero": torch.zeros(r108_B, 192),
    "empty_encoder": empty_set_vector("encode_action", r108_B),
}
r108_results = []
r108_best = None
for ek in range(rm_W["embed.ekind.weight"].shape[0]):
    for gu in range(rm_W["embed.categorical.unit.weight"].shape[0]):
        for mu in range(rm_W["embed.categorical.unit.weight"].shape[0]):
            units = torch.where(
                torch.as_tensor(r103_is_glucose, dtype=torch.bool),
                torch.tensor(gu), torch.tensor(mu))
            atoms = (r103_base + rm_W["embed.ekind.weight"][ek]
                     + rm_W["embed.categorical.unit.weight"][units])
            encoded = rm_mlp(atoms, "encode.mlp1")
            pooled = torch.zeros(r108_B, 192)
            pooled.index_add_(0, r108_bundle_index, encoded)
            e = rm_ln(rm_mlp(pooled / r108_count.sqrt() + r108_size,
                            "encode.mlp2"), "encode.norm")
            for action_name, u in r108_actions.items():
                z = rm_mlp(torch.cat([r103_ctx, e, u], dim=1), "z_post")[:, :32]
                error = float((z - r103_target).square().mean().sqrt())
                spec = (ek, gu, mu, action_name)
                r108_results.append((error, spec))
                if r108_best is None or error < r108_best["rmse"]:
                    r108_best = {"rmse": error, "spec": spec,
                                 "e": e.clone(), "u": u.clone(), "z": z.clone()}

print("Best posterior candidates:")
for item in sorted(r108_results)[:10]:
    print(item)
with open("/work/notes.md", "a") as f:
    f.write("\n## Empty-action diagnostic; Python 108, forward 11/50\n"
            f"- Best measurement posterior: {r108_best['rmse']:.9g}, "
            f"spec={r108_best['spec']}.\n"
            "- No weight edits; synthetic transition work still requires exact validation.\n")
```
