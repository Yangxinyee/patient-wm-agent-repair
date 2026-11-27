Correcting empty actions left a smaller measurement error, and changing unit IDs did not resolve it. I’ll make one targeted check of omitted categorical fields and per-sensor units. If that fails, I’ll proceed using the already-validated action bundles rather than let measurement reconstruction delay the repair indefinitely.

```python
base_emb59 = observation_embeddings56(gu=1, mu=2, me=1)
measurement_atom59 = np.isin(obs_codes51, [
    "glucose_finger", "glucose_lab", "map_arterial", "map_cuff"])
measure_tensor59 = torch.tensor(measurement_atom59, dtype=torch.float32)[:, None]

def eval_emb59(emb):
    e = encode_set50(emb, obs_owner51, obs_counts51, "encode")
    h = F.gelu(affine42(torch.cat(
        [ctx51, e[valid_indices51], u_correct58[valid_indices51]], dim=1),
        "z_post.0"))
    z = affine42(h, "z_post.2")[:, :32]
    return float(torch.mean((z - target_z51)**2).sqrt()), e, h, z

cat_search59 = [("base", 0, eval_emb59(base_emb59)[0])]
for tensor59 in ["embed.kq.weight"] + [
    n for n in NAMES if n.startswith("embed.categorical.")
    and "unit" not in n]:
    w59 = wt42(tensor59)
    print(tensor59, tuple(w59.shape))
    for idx59 in range(1, w59.shape[0]):
        candidate59 = base_emb59 + measure_tensor59 * (w59[idx59] - w59[0])
        cat_search59.append((tensor59, idx59, eval_emb59(candidate59)[0]))
print("Categorical tests:", sorted(cat_search59, key=lambda x: x[2])[:12])

unit_by_code59 = {}
emb_work59 = base_emb59.clone()
for pass59 in range(2):
    for code59 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        old59 = unit_by_code59.get(
            code59, 1 if code59.startswith("glucose") else 2)
        mask59 = torch.tensor(obs_codes51 == code59, dtype=torch.float32)[:, None]
        options59 = []
        for idx59 in range(wt42("embed.categorical.unit.weight").shape[0]):
            cand59 = emb_work59 + mask59 * (
                wt42("embed.categorical.unit.weight")[idx59]
                - wt42("embed.categorical.unit.weight")[old59])
            options59.append((eval_emb59(cand59)[0], idx59))
        score59, new59 = min(options59)
        emb_work59 = emb_work59 + mask59 * (
            wt42("embed.categorical.unit.weight")[new59]
            - wt42("embed.categorical.unit.weight")[old59])
        unit_by_code59[code59] = new59
print("Per-code units:", unit_by_code59, "RMS:", eval_emb59(emb_work59)[0])

print("\nContext calculation:")
for ins59 in dis.get_instructions(contexts37):
    print(ins59.opname, ins59.argrepr)
print("\nHead/tensor dimensions:")
for n59 in NAMES:
    if n59.startswith(("z_post", "z_in", "z_head", "attr_heads.value_bin",
                       "last_numeric", "core.prelude.0")):
        print(n59, SHAPES[n59])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 59 / targeted remaining-encoder audit\n"
            f"Measurement categorical candidates: {sorted(cat_search59, key=lambda x:x[2])[:4]}; "
            f"per-code unit fit {unit_by_code59}. Action-only local posterior "
            "already validated; no tensors edited.\n")
```
