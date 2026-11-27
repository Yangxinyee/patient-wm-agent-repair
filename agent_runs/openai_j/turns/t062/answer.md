The action embeddings are now specified, and the stored dose innovations align with the following hour’s readings. Most action bundles also contain infusion END events, so those cannot be ignored. I’ll reconstruct the posterior features by bundle type; small errors on mixed bundles would make them suitable for a selective edit.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF

torch.set_num_threads(8)
repair_states = latent_probe_state
repair_sf = state_frame.copy()
assert np.array_equal(repair_sf.stay_id.to_numpy(), repair_states["stay_id"])
repair_n = len(repair_sf)
repair_world = train[train.stay_id.isin(repair_sf.stay_id.unique())].copy()
repair_world["is_end"] = False
repair_ends = repair_world[repair_world.end_hour.notna()].copy()
repair_ends["hour"] = repair_ends["end_hour"]
repair_ends["is_end"] = True
repair_atoms = pd.concat([repair_world, repair_ends], ignore_index=True)
repair_atoms["hour_key"] = repair_atoms.hour.round(4)
repair_sf["hour_key"] = repair_sf.hour.round(4)
repair_atoms = repair_atoms.merge(
    repair_sf[["stay_id", "hour_key", "state_index"]],
    on=["stay_id", "hour_key"], how="inner"
).sort_values("state_index").reset_index(drop=True)

counts_check = repair_atoms.groupby("state_index").size().reindex(
    range(repair_n), fill_value=0).to_numpy()
print("Atom-count mismatches:", int(np.sum(counts_check != repair_sf.n_atoms.to_numpy())))
print("Numeric convention:", VOCAB["numeric_note"])

repair_codes = {x["name"]: x["id"] for x in VOCAB["codes"]}
repair_norms = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
repair_units = {"glucose_finger": 1, "glucose_lab": 1,
                "map_arterial": 2, "map_cuff": 2,
                "insulin": 3, "norepinephrine": 4}
repair_runits = {"insulin": 2, "norepinephrine": 3}
repair_ekinds = {"glucose_finger": 1, "glucose_lab": 1,
                 "map_arterial": 1, "map_cuff": 1,
                 "insulin": 2, "norepinephrine": 2,
                 "hospital_admission": 3, "icu_admission": 4}
repair_is_end = repair_atoms.is_end.to_numpy()
repair_is_action = (
    repair_atoms.event.isin(["insulin", "norepinephrine"]).to_numpy()
    & ~repair_is_end
)
repair_bi = torch.as_tensor(repair_atoms.state_index.to_numpy(), dtype=torch.long)

def repair_atom_vectors(end_numbers=False, end_units=False):
    codes = torch.tensor([repair_codes[x] for x in repair_atoms.event])
    kinds = torch.tensor(np.where(repair_is_end, 2, 0))
    ekinds = torch.tensor([repair_ekinds[x] for x in repair_atoms.event])
    units = np.array([repair_units.get(x, 0) for x in repair_atoms.event])
    runits = np.array([repair_runits.get(x, 0) for x in repair_atoms.event])
    if not end_units:
        units[repair_is_end] = 0
        runits[repair_is_end] = 0
    result = (wt("embed.code.weight")[codes]
              + wt("embed.kind.weight")[kinds]
              + wt("embed.ekind.weight")[ekinds])
    for table in cat_names:
        idx = (units if table == "embed.categorical.unit.weight" else
               runits if table == "embed.categorical.rate_unit.weight" else
               np.zeros(len(repair_atoms), dtype=int))
        result = result + wt(table)[torch.as_tensor(idx, dtype=torch.long)]
    numbers = np.zeros((len(repair_atoms), 6), dtype=np.float32)
    for attr_i, attr in enumerate(["value", "dose", "rate"]):
        bins = np.zeros(len(repair_atoms), dtype=np.int64)
        for event in repair_atoms.event.unique():
            norm = repair_norms.get((event, attr))
            if norm is None:
                continue
            values = repair_atoms[attr].to_numpy(dtype=float)
            mask = ((repair_atoms.event.to_numpy() == event) & np.isfinite(values))
            if not end_numbers:
                mask &= ~repair_is_end
            numbers[mask, 2*attr_i] = np.clip(
                (values[mask] - norm["median"]) / norm["iqr"], -10, 10)
            numbers[mask, 2*attr_i+1] = 1
            bins[mask] = np.searchsorted(
                norm["fine_bin_edges"], values[mask], side="left") + 1
        result = result + wt("embed.bins." + attr + ".weight")[torch.tensor(bins)]
    return result + TF.linear(torch.tensor(numbers),
                              wt("embed.numeric.weight"), wt("embed.numeric.bias"))

def repair_set_encode(atom_vectors, mask, prefix, zero_empty=False):
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    h_atoms = mlp(atom_vectors[mask_t], prefix + ".mlp1", "gelu")
    pooled = torch.zeros((repair_n, 192))
    pooled.index_add_(0, repair_bi[mask_t], h_atoms)
    sizes = torch.bincount(repair_bi[mask_t], minlength=repair_n).float()
    pooled = pooled / sizes.clamp_min(1).sqrt()[:, None]
    pooled += TF.linear(torch.log1p(sizes)[:, None],
                        wt(prefix + ".size.weight"), wt(prefix + ".size.bias"))
    result = ln(mlp(pooled, prefix + ".mlp2", "gelu"), prefix + ".norm")
    if zero_empty:
        result[sizes == 0] = 0
    return result

repair_context = torch.zeros((repair_n, 192))
repair_context[1:] = latent_context_at(np.arange(1, repair_n), repair_states)
repair_valid = np.r_[False, np.diff(repair_sf.stay_id.to_numpy()) == 0]
repair_has_end = np.bincount(
    repair_atoms.state_index.to_numpy()[repair_is_end], minlength=repair_n) > 0
repair_has_action = (repair_sf.has_insulin | repair_sf.has_norepinephrine).to_numpy()
repair_truth_z = torch.as_tensor(repair_states["x"][:, 960:992])

repair_variant_scores = []
repair_features = {}
for end_numbers in [False, True]:
    for end_units in [False, True]:
        av = repair_atom_vectors(end_numbers, end_units)
        ev = repair_set_encode(av, ~repair_is_action, "encode")
        uv = repair_set_encode(av, repair_is_action, "encode_action", zero_empty=True)
        posterior_input = torch.cat([repair_context, ev, uv], dim=1)
        hidden = activation(TF.linear(posterior_input, wt("z_post.0.weight"),
                                     wt("z_post.0.bias")), "gelu")
        predicted_z = TF.linear(hidden, wt("z_post.2.weight"),
                                wt("z_post.2.bias"))[:, :32]
        error = ((predicted_z - repair_truth_z)**2).mean(1).numpy()
        label = (end_numbers, end_units)
        repair_features[label] = hidden.numpy()
        row = {"end_numbers": end_numbers, "end_units": end_units}
        for group_name, group_mask in [
            ("all", repair_valid),
            ("action_no_end", repair_valid & repair_has_action & ~repair_has_end),
            ("action_with_end", repair_valid & repair_has_action & repair_has_end),
            ("observation_no_end", repair_valid & ~repair_has_action & ~repair_has_end),
            ("end_no_action", repair_valid & ~repair_has_action & repair_has_end)
        ]:
            row[group_name] = float(np.sqrt(error[group_mask].mean()))
        repair_variant_scores.append(row)
print(pd.DataFrame(repair_variant_scores).to_string(index=False))
print("Edits:", edited())
```
