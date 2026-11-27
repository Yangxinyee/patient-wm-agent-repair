Measurement bundles still do not match, so we cannot yet trust a complete local update. The saved flags also appear to mark infusion starts, not all infusion-end atoms. I’ll test how empty action sets and END atoms are encoded; matching both drug and measurement bundles is the required check.

```python
empty_action_local = norm0(
    mlp0(lin0(torch.zeros(1, 1), "encode_action.size"),
         "encode_action.mlp2", "gelu"), "encode_action.norm")
empty_measure_local = norm0(
    mlp0(lin0(torch.zeros(1, 1), "encode.size"),
         "encode.mlp2", "gelu"), "encode.norm")

for drug, probe in action_probe_data.items():
    ctx, _, _ = context_from_saved(probe["rows"])
    u = set_encode_single_local(
        action_atom_exact(drug, probe["rates"], action_probe_exact_doses[drug]),
        "encode_action")
    zp = mlp0(torch.cat([ctx, empty_measure_local.expand(len(ctx), -1), u], 1),
              "z_post", "gelu")[:, :32]
    target_z = torch.from_numpy(state4["x"][probe["rows"], 960:])
    print(drug, "unrounded-time latent RMSE", float((zp-target_z).square().mean().sqrt()))

measurement_reconstruction_trials = {}
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    rows, values = [], []
    for j in range(1, len(state4["hour"])):
        key = (int(state4["stay_id"][j]), int(round(state4["hour"][j]*60)))
        recs = record_lookup_local.get(key, [])
        if state4["stay_id"][j] == state4["stay_id"][j-1] and \
                state4["n_atoms"][j] == 1 and len(recs) == 1 and recs[0].event == event:
            rows.append(j); values.append(recs[0].value)
    rows, values = np.asarray(rows[:150]), np.asarray(values[:150])
    ctx, _, _ = context_from_saved(rows)
    target_z = torch.from_numpy(state4["x"][rows, 960:])
    trials = []
    for unit, quality, empty_u in product(range(6), (-1, 0), ("zero", "encoded")):
        atom = measurement_atom_local(event, values, unit)
        if quality == -1:
            atom -= W0["embed.kq.weight"][0]
        e = set_encode_single_local(atom, "encode")
        u = (torch.zeros_like(e) if empty_u == "zero"
             else empty_action_local.expand(len(e), -1))
        zp = mlp0(torch.cat([ctx, e, u], 1), "z_post", "gelu")[:, :32]
        trials.append((float((zp-target_z).square().mean().sqrt()), unit, quality, empty_u))
    trials.sort()
    measurement_reconstruction_trials[event] = trials
    print(event, "best", trials[:3])

for drug in ("insulin", "norepinephrine"):
    rows, recs = [], []
    for j in range(1, len(state4["hour"])):
        key = (int(state4["stay_id"][j]), int(round(state4["hour"][j]*60)))
        endkey = (*key, drug)
        if state4["stay_id"][j] == state4["stay_id"][j-1] and \
                state4["n_atoms"][j] == 1 and not record_lookup_local.get(key) and \
                endkey in end_lookup_local:
            rows.append(j); recs.append(end_lookup_local[endkey])
    rows = np.asarray(rows[:120]); recs = recs[:120]
    if not len(rows):
        print(drug, "no END probes"); continue
    ctx, _, _ = context_from_saved(rows)
    target_z = torch.from_numpy(state4["x"][rows, 960:])
    trials = []
    for numbers, units, destination in product((False, True), (False, True), ("e", "u")):
        if numbers:
            atom = action_atom_exact(drug, [r.rate for r in recs], [r.dose for r in recs], kind=2)
        else:
            atom = (W0["embed.code.weight"][code_id[drug]] + W0["embed.kind.weight"][2]
                    + W0["embed.ekind.weight"][2] + W0["embed.kq.weight"][0]
                    + W0["embed.numeric.bias"]).expand(len(rows), -1).clone()
            atom += W0["embed.categorical.unit.weight"][3 if drug == "insulin" else 4]
            atom += W0["embed.categorical.rate_unit.weight"][2 if drug == "insulin" else 3]
        if not units:
            atom -= W0["embed.categorical.unit.weight"][3 if drug == "insulin" else 4]
            atom -= W0["embed.categorical.rate_unit.weight"][2 if drug == "insulin" else 3]
        e = empty_measure_local.expand(len(rows), -1)
        u = empty_action_local.expand(len(rows), -1)
        if destination == "e":
            e = set_encode_single_local(atom, "encode")
        else:
            u = set_encode_single_local(atom, "encode_action")
        zp = mlp0(torch.cat([ctx, e, u], 1), "z_post", "gelu")[:, :32]
        trials.append((float((zp-target_z).square().mean().sqrt()), numbers, units, destination))
    trials.sort()
    end_convention_trials[drug] = trials
    print(drug, "END n", len(rows), "best", trials[:4])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 27–33: local encoder reconstruction\n"
            "- Confirmed state layout: out = core.out_norm(mean of four h slots).\n"
            "- Drug-start latent reconstruction uses GELU, quality embedding row 0, and encoded empty measurement set.\n"
            "- Insulin: dose unit 3, rate unit 2; NE: dose unit 4, rate unit 3.\n"
            "- Recorded decimal doses resolve NE bin-boundary discrepancies; left-sided binning confirmed.\n"
            "- Measurement/empty-action and infusion-END conventions are being checked before any recurrence edit.\n"
            "- No tensors edited; 6 forward passes used.\n")
```
