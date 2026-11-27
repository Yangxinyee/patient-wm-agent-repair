The best matches are not yet close enough to trust: posterior RMSE is 0.164 for insulin and 0.083 for norepinephrine. GELU and the documented bin convention fit best, but an embedding term or empty-bundle convention may be missing. I’ll test those possibilities. Errors near numerical precision would confirm the reconstruction.

```python
print("=== ADDITIONAL EMBEDDING ROW NORMS ===")
for name in ("embed.kq.weight", "embed.text.weight",
             "embed.categorical.unit.weight", "embed.categorical.rate_unit.weight"):
    print(name, np.round(torch.linalg.vector_norm(W0[name], dim=1).numpy(), 4))

def recon_single_set(atoms, prefix="encode_action"):
    pooled = recon_mlp(atoms, prefix+".mlp1")
    pooled = pooled + recon_lin(
        torch.full((*pooled.shape[:-1], 1), np.log(2), dtype=torch.float32),
        prefix+".size"
    )
    return recon_norm(recon_mlp(pooled, prefix+".mlp2"), prefix+".norm")

def recon_atom(drug, records, unit_id=0, rate_unit_id=0, kq_id=None):
    numeric = np.zeros((len(records), 6), dtype=np.float32)
    bins = {}
    for col, attr in ((2, "dose"), (4, "rate")):
        values = np.array([getattr(r, attr) for r in records])
        spec = NUM_SPEC[(drug, attr)]
        numeric[:, col] = np.clip((values-spec["median"])/spec["iqr"], -10, 10)
        numeric[:, col+1] = 1
        bins[attr] = np.searchsorted(spec["fine_bin_edges"], values, side="left")+1
    atom = (
        W0["embed.code.weight"][CODE_ID[drug]] +
        W0["embed.kind.weight"][0] + W0["embed.ekind.weight"][2] +
        F.linear(torch.from_numpy(numeric), W0["embed.numeric.weight"],
                 W0["embed.numeric.bias"]) +
        W0["embed.bins.dose.weight"][bins["dose"]] +
        W0["embed.bins.rate.weight"][bins["rate"]] +
        W0["embed.categorical.unit.weight"][unit_id] +
        W0["embed.categorical.rate_unit.weight"][rate_unit_id]
    )
    if kq_id is not None:
        atom = atom + W0["embed.kq.weight"][kq_id]
    return atom

empty_e = recon_norm(
    recon_mlp(recon_lin(torch.zeros(1, 1), "encode.size"), "encode.mlp2"),
    "encode.norm"
)
expanded_recon_search = []
extended_choices = list(itertools.product(range(6), range(4), [None, 0, 1, 2, 3]))

for drug, (indices, records) in pure_action_rows.items():
    if not len(indices):
        continue
    atoms = torch.stack([recon_atom(drug, records, ui, ri, ki)
                         for ui, ri, ki in extended_choices])
    uu = recon_single_set(atoms)
    ee_action = recon_single_set(atoms, "encode")
    ctx = recon_ctx(indices)
    target = torch.as_tensor(S0["x"][indices, 960:992])
    prior_mean = recon_mlp(ctx, "z_prior")[:, :32]
    print(drug, "saved posterior RMS:", float(target.square().mean().sqrt()),
          "prior mean RMS:", float(prior_mean.square().mean().sqrt()))
    for e_mode in ("zero", "empty", "action"):
        if e_mode == "zero":
            ee = torch.zeros_like(uu)
        elif e_mode == "empty":
            ee = empty_e.expand_as(uu)
        else:
            ee = ee_action
        xx = torch.cat((ctx.expand(len(extended_choices), -1, -1), ee, uu), dim=-1)
        zp = recon_mlp(xx, "z_post")[..., :32]
        for add_prior in (False, True):
            zz = zp + prior_mean[None] if add_prior else zp
            errors = (zz-target[None]).square().mean((1, 2)).sqrt()
            for ci, (ui, ri, ki) in enumerate(extended_choices):
                expanded_recon_search.append({
                    "drug": drug, "rmse": float(errors[ci]),
                    "unit_id": ui, "rate_unit_id": ri, "kq_id": ki,
                    "e_mode": e_mode, "add_prior": add_prior
                })

expanded_recon_best = {}
print("\n=== EXTENDED BEST MATCHES ===")
for drug in DRUGS:
    matches = sorted([r for r in expanded_recon_search if r["drug"] == drug],
                     key=lambda r: r["rmse"])
    if matches:
        expanded_recon_best[drug] = matches[0]
        for result in matches[:4]:
            print(result)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 18: reconstruction error audit\n"
        "Single-start posterior errors remain too large for a verified reconstruction. "
        "Tested key-quality embeddings, empty observation encoding, dual encoding, "
        "and possible prior-mean addition. Best candidates: "
        + repr(expanded_recon_best) +
        "\nNo weights edited. Python runs: 18. Forward requests: 3.\n"
    )
```