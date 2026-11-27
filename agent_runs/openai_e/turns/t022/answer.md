The posterior now matches to about \(4\times10^{-5}\): empty observation bundles have a nonzero encoding, and atoms include a key-quality embedding. I’ll extend the reconstruction to infusion endings and all saved bundles. Agreement across the sequence would support using local computations to design the edit; remaining mismatches would identify conventions still missing.

```python
RECON_UNITS = {
    "glucose_finger": 2, "glucose_lab": 2,
    "map_arterial": 5, "map_cuff": 5,
    "insulin": 3, "norepinephrine": 4,
    "hospital_admission": 0, "icu_admission": 0
}
RECON_RATE_UNITS = {"insulin": 2, "norepinephrine": 3}
RECON_EKINDS = {
    item["name"]: {"measurement": 1, "drug_admin": 2, "visit": 3, "visit_detail": 4}[item["event_kind"]]
    for item in VOCAB["codes"] if "event_kind" in item
}

print("=== EXACT GAP CONVENTION ===")
gap_errors = {}
for style in ("log", "log1p"):
    errors = []
    for drug, (indices, records) in pure_action_rows.items():
        if not len(indices):
            continue
        atom = recon_atom(drug, records, RECON_UNITS[drug],
                          RECON_RATE_UNITS[drug], 0)
        uu = recon_single_set(atom)
        ctx = recon_ctx(indices, gap_style=style)
        zz = recon_mlp(torch.cat((ctx, empty_e.expand_as(uu), uu), -1), "z_post")[:, :32]
        errors.append((zz-torch.as_tensor(S0["x"][indices, 960:992])).square().mean())
    gap_errors[style] = float(torch.stack(errors).mean().sqrt())
    print(style, gap_errors[style])
RECON_GAP = min(gap_errors, key=gap_errors.get)

def recon_record_atom(row, kind=0, end_numeric="absent", end_units="none"):
    drug = row.event
    numeric = torch.zeros(6)
    bin_terms = torch.zeros(192)
    for col, attr in ((0, "value"), (2, "dose"), (4, "rate")):
        val = getattr(row, attr)
        if not np.isfinite(val) or (kind == 2 and end_numeric == "absent"):
            continue
        if kind == 2 and end_numeric == "zero":
            val = 0.0
        spec = NUM_SPEC.get((drug, attr))
        if spec is None:
            continue
        numeric[col] = float(np.clip((val-spec["median"])/spec["iqr"], -10, 10))
        numeric[col+1] = 1
        bb = int(np.searchsorted(spec["fine_bin_edges"], val, side="left")+1)
        bin_terms += W0["embed.bins."+attr+".weight"][bb]
    ui = RECON_UNITS[drug] if kind == 0 or end_units in ("code", "all") else 0
    ri = RECON_RATE_UNITS.get(drug, 0) if kind == 0 or end_units == "all" else 0
    return (
        W0["embed.code.weight"][CODE_ID[drug]] +
        W0["embed.kind.weight"][kind] +
        W0["embed.ekind.weight"][RECON_EKINDS[drug]] +
        W0["embed.kq.weight"][0] +
        W0["embed.categorical.unit.weight"][ui] +
        W0["embed.categorical.rate_unit.weight"][ri] +
        F.linear(numeric, W0["embed.numeric.weight"], W0["embed.numeric.bias"]) +
        bin_terms
    )

def recon_sets(atom_lists, prefix="encode", empty_zero=False):
    counts = torch.tensor([len(x) for x in atom_lists], dtype=torch.float32)
    width = max(1, max(len(x) for x in atom_lists))
    packed = torch.zeros(len(atom_lists), width, 192)
    mask = torch.zeros(len(atom_lists), width, 1)
    for i, atoms in enumerate(atom_lists):
        if atoms:
            packed[i, :len(atoms)] = torch.stack(atoms)
            mask[i, :len(atoms)] = 1
    pooled = (recon_mlp(packed, prefix+".mlp1")*mask).sum(1)
    pooled /= counts.clamp_min(1).sqrt()[:, None]
    pooled += recon_lin(torch.log1p(counts)[:, None], prefix+".size")
    result = recon_norm(recon_mlp(pooled, prefix+".mlp2"), prefix+".norm")
    if empty_zero:
        result *= (counts > 0)[:, None]
    return result

end_indices = []
for j in np.flatnonzero(S0["has_insulin"] | S0["has_norepinephrine"]):
    key = (int(S0["stay_id"][j]), int(round(S0["hour"][j]*60)))
    ns, ne = len(drug_starts.get(key, [])), len(drug_ends.get(key, []))
    if (ne and ns+ne == int(S0["n_atoms"][j]) and j > 0
        and S0["stay_id"][j-1] == S0["stay_id"][j]):
        end_indices.append(j)
end_indices = np.array(end_indices[::max(1, len(end_indices)//80)][:80])
end_ctx = recon_ctx(end_indices, gap_style=RECON_GAP)
end_target = torch.as_tensor(S0["x"][end_indices, 960:992])
end_search = []

for end_numeric, end_units in itertools.product(("absent", "copy", "zero"),
                                                ("none", "code", "all")):
    lists = []
    for j in end_indices:
        key = (int(S0["stay_id"][j]), int(round(S0["hour"][j]*60)))
        atoms = [recon_record_atom(r) for r in drug_starts.get(key, [])]
        atoms += [recon_record_atom(r, 2, end_numeric, end_units)
                  for r in drug_ends.get(key, [])]
        lists.append(atoms)
    uu = recon_sets(lists, "encode_action", empty_zero=True)
    zz = recon_mlp(torch.cat((end_ctx, empty_e.expand_as(uu), uu), -1), "z_post")[:, :32]
    end_search.append({
        "end_numeric": end_numeric, "end_units": end_units,
        "rmse": float((zz-end_target).square().mean().sqrt())
    })
END_CFG = min(end_search, key=lambda x: x["rmse"])
print("\n=== END ENCODING ===")
for result in sorted(end_search, key=lambda x: x["rmse"])[:3]:
    print(result)

record_starts, record_ends = {}, {}
for row in tune.itertuples(index=False):
    record_starts.setdefault((int(row.stay_id), int(round(row.hour*60))), []).append(row)
    if np.isfinite(row.end_hour):
        record_ends.setdefault((int(row.stay_id), int(round(row.end_hour*60))), []).append(row)

recon_obs_lists, recon_action_lists, recon_atom_counts = [], [], []
for j in range(len(S0["x"])):
    key = (int(S0["stay_id"][j]), int(round(S0["hour"][j]*60)))
    obs, acts = [], []
    for r in record_starts.get(key, []):
        (acts if r.event in DRUGS else obs).append(recon_record_atom(r))
    for r in record_ends.get(key, []):
        atom = recon_record_atom(r, 2, END_CFG["end_numeric"], END_CFG["end_units"])
        (acts if r.event in DRUGS else obs).append(atom)
    recon_obs_lists.append(obs)
    recon_action_lists.append(acts)
    recon_atom_counts.append(len(obs)+len(acts))

recon_e = recon_sets(recon_obs_lists)
recon_u = recon_sets(recon_action_lists, "encode_action", empty_zero=True)
recon_valid_rows = np.flatnonzero(
    np.r_[False, S0["stay_id"][1:] == S0["stay_id"][:-1]]
)
recon_context = torch.zeros(len(S0["x"]), 192)
recon_context[recon_valid_rows] = recon_ctx(recon_valid_rows, gap_style=RECON_GAP)
recon_post_inputs = torch.cat((recon_context, recon_e, recon_u), -1)
recon_post_hidden = F.gelu(recon_lin(recon_post_inputs, "z_post.0"))
recon_post_mean = recon_lin(recon_post_hidden, "z_post.2")[:, :32]
recon_errors = (
    recon_post_mean-torch.as_tensor(S0["x"][:, 960:992])
).square().mean(1).sqrt().numpy()
count_mismatch = np.array(recon_atom_counts) != S0["n_atoms"]

print("\n=== WHOLE-SEQUENCE RECONSTRUCTION ===")
print("atom-count mismatches:", int(count_mismatch.sum()))
action_flag = S0["has_insulin"] | S0["has_norepinephrine"]
for label, mask in (("action", action_flag), ("non-action", ~action_flag)):
    rr = recon_valid_rows[mask[recon_valid_rows]]
    print(label, "bundles:", len(rr), "RMSE:", float(np.sqrt(np.mean(recon_errors[rr]**2))),
          "max bundle RMSE:", float(recon_errors[rr].max()))
for j in recon_valid_rows[np.argsort(recon_errors[recon_valid_rows])[-3:]]:
    print("worst:", int(S0["stay_id"][j]), float(S0["hour"][j]),
          "error", float(recon_errors[j]), "atom count", recon_atom_counts[j],
          "reported", int(S0["n_atoms"][j]))
print("training stay IDs:", df.stay_id.drop_duplicates().head(5).tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 19: END and whole-sequence reconstruction\n"
        "Single starts match after including kq[0] and nonzero empty observation encoding. "
        f"Gap={RECON_GAP}; END={END_CFG}; atom-count mismatches={int(count_mismatch.sum())}. "
        f"Whole-sequence posterior RMSE={np.sqrt(np.mean(recon_errors[recon_valid_rows]**2)):.7f}.\n"
        "No weights edited. Python runs: 19. Forward requests: 3.\n"
    )
```