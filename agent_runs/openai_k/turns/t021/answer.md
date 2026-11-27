The helpers use standard linear layers and normalization, but the reconstruction has not isolated empty-bundle handling or all embedding conventions. I’ll test those against the saved posterior means. A near-exact match with consistent conventions across both drugs would validate the implementation; a large residual would mean another component is still missing.

```python
import itertools, time
torch.set_num_threads(8)
audit_start = time.time()

# Match single-action bundles to their recorded infusion atoms.
audit_ids = np.unique(train_states0["stay_id"][first_drug0])
audit_records = df[df.stay_id.isin(audit_ids) & df.event.isin(drug_codes)]
audit_lookup = {
    (int(row.stay_id), round(float(row.hour), 4)): row
    for row in audit_records.itertuples(index=False)
}

def encode_single_audit(atom, prefix, activation):
    hidden = mlp0(atom, prefix + ".mlp1", activation)
    size = lin0(torch.full((*atom.shape[:-1], 1), np.log(2), dtype=torch.float32),
                prefix + ".size")
    return norm0(mlp0(hidden + size, prefix + ".mlp2", activation),
                 prefix + ".norm")

empty_world_audit = {
    "zero": torch.zeros(192),
    **{
        "empty_" + activation: norm0(
            mlp0(lin0(torch.zeros(1, 1), "encode.size"),
                 "encode.mlp2", activation), "encode.norm")[0]
        for activation in ("gelu", "silu", "relu")
    }
}

audit_options = list(itertools.product(
    ("left", "right"), (-1, 0, 1, 2, 3), range(6), range(4),
    ("none", "action", "status_action", "all")
))
audit_best = {}

for drug in drug_codes:
    indices = np.array([
        int(i) for i in first_drug0
        if (int(train_states0["stay_id"][i]), round(float(train_states0["hour"][i]), 4))
           in audit_lookup
        and audit_lookup[(int(train_states0["stay_id"][i]),
                          round(float(train_states0["hour"][i]), 4))].event == drug
    ])
    rows = [audit_lookup[(int(train_states0["stay_id"][i]),
                         round(float(train_states0["hour"][i]), 4))] for i in indices]
    previous = torch.tensor(sx[indices - 1])
    target_audit = torch.tensor(sx[indices, 960:992])
    dt_audit = torch.tensor(
        (train_states0["hour"][indices] - train_states0["hour"][indices - 1]) * 3600,
        dtype=torch.float32
    ).clamp_min(1)
    ph = previous[:, 192:960].reshape(-1, 4, 192)
    fh = ph * torch.exp(-dt_audit[:, None, None] *
                       torch.exp(W0["core.flow.a"])[None, None, :])
    context_base = previous[:, :192] + norm0(fh.mean(1), "core.out_norm")
    print(drug, "n =", len(indices), "out/h consistency RMS =",
          float((norm0(ph.mean(1), "core.out_norm") - previous[:, :192])
                .square().mean().sqrt()))

    numbers = np.zeros((len(rows), 6), np.float32)
    base_bins = {}
    for attribute, zcol, hascol in (("dose", 2, 3), ("rate", 4, 5)):
        values = np.array([getattr(row, attribute) for row in rows])
        spec = num_spec0[(drug, attribute)]
        numbers[:, zcol] = np.clip((values - spec["median"]) / spec["iqr"], -10, 10)
        numbers[:, hascol] = 1
        for side in ("left", "right"):
            base_bins[(attribute, side)] = torch.tensor(
                1 + np.searchsorted(spec["fine_bin_edges"], values, side=side),
                dtype=torch.long
            )

    common_atom = (W0["embed.code.weight"][code_id0[drug]]
                   + W0["embed.kind.weight"][0]
                   + W0["embed.ekind.weight"][2]
                   + lin0(torch.tensor(numbers), "embed.numeric"))
    atoms = []
    for side, kq, unit, rate_unit, defaults in audit_options:
        atom = (common_atom
                + W0["embed.bins.dose.weight"][base_bins[("dose", side)]]
                + W0["embed.bins.rate.weight"][base_bins[("rate", side)]]
                + W0["embed.categorical.unit.weight"][unit]
                + W0["embed.categorical.rate_unit.weight"][rate_unit])
        if kq >= 0:
            atom = atom + W0["embed.kq.weight"][kq]
        fields = {
            "none": (), "action": ("action",),
            "status_action": ("status", "action"),
            "all": ("status", "action", "route", "discharged_to")
        }[defaults]
        for field in fields:
            atom = atom + W0["embed.categorical." + field + ".weight"][1]
        atoms.append(atom)
    atoms = torch.stack(atoms)

    contexts = []
    for gap_activation, gap_input in itertools.product(
            ("gelu", "silu", "relu"), ("log1p/10", "log/10", "log1p", "log")):
        gx = torch.log1p(dt_audit) if "log1p" in gap_input else torch.log(dt_audit)
        if "/10" in gap_input:
            gx = gx / 10
        contexts.append(((gap_activation, gap_input),
                         context_base + mlp0(gx[:, None], "gap", gap_activation)))

    candidates_audit = []
    pw = W0["z_post.0.weight"]
    for encoder_activation in ("gelu", "silu", "relu"):
        actions = encode_single_audit(atoms, "encode_action", encoder_activation)
        action_part = F.linear(actions, pw[:, 384:])
        for (gap_activation, gap_input), context in contexts:
            context_part = F.linear(context, pw[:, :192], W0["z_post.0.bias"])
            for empty_label, empty_e in empty_world_audit.items():
                preactivation = (action_part + context_part[None]
                                 + F.linear(empty_e, pw[:, 192:384]))
                for post_activation in ("gelu", "silu", "relu"):
                    raw_post = lin0(act0(preactivation, post_activation), "z_post.2")
                    for layout in ("chunk", "interleaved"):
                        means = raw_post[..., :32] if layout == "chunk" else raw_post[..., ::2]
                        errors = (means - target_audit[None]).square().mean((1, 2)).sqrt()
                        best_indices = torch.topk(errors, 2, largest=False).indices.tolist()
                        for j in best_indices:
                            candidates_audit.append((
                                float(errors[j]), encoder_activation, gap_activation,
                                gap_input, empty_label, post_activation, layout,
                                audit_options[j]
                            ))
    audit_best[drug] = sorted(candidates_audit)[:12]
    print("Best candidates:")
    for candidate in audit_best[drug][:6]:
        print(candidate)

print("Elapsed seconds:", round(time.time() - audit_start, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 16: posterior convention search\n"
            "- Tested strict/non-strict bin indexing, kq and categorical embeddings, empty-world encodings, and independent MLP activations.\n"
            f"- Best reconstruction candidates: {audit_best}\n"
            "- Python runs: 16; forward passes: 4; no weights edited.\n")
```