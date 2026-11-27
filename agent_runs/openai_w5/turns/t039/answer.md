Both drug encoders now reproduce the saved latent means to about 0.00015 RMS; using recorded doses fixed the norepinephrine discrepancy. I’ll validate measurement and infusion-end atoms next, using unrounded time gaps. That will let us reconstruct complete updates rather than isolated action inputs.

```python
def context_from_saved(rows, states=state4):
    before = torch.from_numpy(states["x"][rows-1])
    dt = torch.as_tensor((states["hour"][rows]-states["hour"][rows-1])*3600,
                         dtype=torch.float32)[:, None]
    hf = before[:, 192:960].reshape(-1, 4, 192)
    hf = hf * torch.exp(-dt[:, :, None] * torch.exp(W0["core.flow.a"])[None, None, :])
    ctx = before[:, :192] + norm0(hf.mean(1), "core.out_norm")
    ctx += mlp0(torch.log1p(dt)/10, "gap", "gelu")
    return ctx, hf, dt

def measurement_atom_local(event, values, unit):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    meta = norm_meta[(event, "value")]
    num = torch.zeros(len(values), 6)
    num[:, 0] = torch.as_tensor(np.clip((values-meta["median"])/meta["iqr"], -10, 10),
                                dtype=torch.float32)
    num[:, 1] = 1
    bins = 1 + np.searchsorted(meta["fine_bin_edges"], values, side="left")
    atom = W0["embed.code.weight"][code_id[event]].expand(len(values), -1).clone()
    atom += W0["embed.kind.weight"][0] + W0["embed.ekind.weight"][1]
    atom += W0["embed.kq.weight"][0] + W0["embed.categorical.unit.weight"][unit]
    atom += W0["embed.bins.value.weight"][torch.as_tensor(bins)]
    return atom + lin0(num, "embed.numeric")

record_lookup_local = {}
for row in train[train.stay_id.isin(np.unique(state4["stay_id"]))].itertuples():
    key = (int(row.stay_id), int(round(row.hour*60)))
    record_lookup_local.setdefault(key, []).append(row)

measurement_unit_local = {}
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    rows, values = [], []
    for j in range(1, len(state4["hour"])):
        if state4["stay_id"][j] != state4["stay_id"][j-1] or state4["n_atoms"][j] != 1:
            continue
        key = (int(state4["stay_id"][j]), int(round(state4["hour"][j]*60)))
        records = record_lookup_local.get(key, [])
        if len(records) == 1 and records[0].event == event:
            rows.append(j); values.append(records[0].value)
    rows, values = np.asarray(rows[:150]), np.asarray(values[:150])
    ctx, _, _ = context_from_saved(rows)
    target_z = torch.from_numpy(state4["x"][rows, 960:])
    trials = []
    for unit in range(6):
        e = set_encode_single_local(measurement_atom_local(event, values, unit), "encode")
        zp = mlp0(torch.cat([ctx, e, torch.zeros_like(e)], 1), "z_post", "gelu")[:, :32]
        trials.append((float((zp-target_z).square().mean().sqrt()), unit))
    trials.sort()
    measurement_unit_local[event] = trials[0][1]
    print(event, "n", len(rows), "best", trials[:3])

end_lookup_local = {}
for row in drug_records_probe.itertuples():
    key = (int(row.stay_id), int(round(row.end_hour*60)), row.event)
    end_lookup_local[key] = row

end_convention_trials = {}
for drug, flag in (("insulin", "has_insulin"), ("norepinephrine", "has_norepinephrine")):
    rows, records = [], []
    for j in np.flatnonzero(state4[flag] & (state4["n_atoms"] == 1)):
        key = (int(state4["stay_id"][j]), int(round(state4["hour"][j]*60)), drug)
        if j > 0 and state4["stay_id"][j-1] == state4["stay_id"][j] and \
                key not in drug_lookup_probe and key in end_lookup_local:
            rows.append(j); records.append(end_lookup_local[key])
    rows = np.asarray(rows[:150]); records = records[:150]
    if not len(rows):
        continue
    ctx, _, _ = context_from_saved(rows)
    e = norm0(mlp0(lin0(torch.zeros(len(rows), 1), "encode.size"),
                   "encode.mlp2", "gelu"), "encode.norm")
    target_z = torch.from_numpy(state4["x"][rows, 960:])
    trials = []
    for keep_numbers, keep_units, quality in product((False, True), (False, True), (-1, 0)):
        if keep_numbers:
            atom = action_atom_exact(drug, [r.rate for r in records],
                                      [r.dose for r in records], kind=2)
            if not keep_units:
                atom -= W0["embed.categorical.unit.weight"][3 if drug == "insulin" else 4]
                atom -= W0["embed.categorical.rate_unit.weight"][2 if drug == "insulin" else 3]
            if quality == -1:
                atom -= W0["embed.kq.weight"][0]
        else:
            atom = (W0["embed.code.weight"][code_id[drug]] + W0["embed.kind.weight"][2]
                    + W0["embed.ekind.weight"][2] + W0["embed.numeric.bias"])
            atom = atom.expand(len(rows), -1).clone()
            if keep_units:
                atom += W0["embed.categorical.unit.weight"][3 if drug == "insulin" else 4]
                atom += W0["embed.categorical.rate_unit.weight"][2 if drug == "insulin" else 3]
            if quality == 0:
                atom += W0["embed.kq.weight"][0]
        u = set_encode_single_local(atom, "encode_action")
        zp = mlp0(torch.cat([ctx, e, u], 1), "z_post", "gelu")[:, :32]
        trials.append((float((zp-target_z).square().mean().sqrt()),
                       keep_numbers, keep_units, quality))
    trials.sort()
    end_convention_trials[drug] = trials
    print(drug, "single END n", len(rows), "best", trials[:4])
```
