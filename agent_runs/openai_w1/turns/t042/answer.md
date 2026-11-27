Routing END atoms through the observation encoder reduced the error from 0.139 to 0.020, and GELU now fits best throughout. That is encouraging but not yet exact. I’ll check END metadata and pooling counts; numerical agreement would let me build the edit from validated internal features rather than approximations.

```python
# Diagnose whether the remaining mismatch is concentrated in END bundles.
r35_diag = []
for r35_j, r35_i in enumerate(r33_sample):
    r35_key = (int(r33_ids[r35_i]), round(float(r33_hours[r35_i]), 4))
    r35_diag.append({
        "n_start": len(r33_starts.get(r35_key, [])),
        "n_end": len(r33_ends.get(r35_key, [])),
        "insulin": any(a[0] == "insulin" for a in r33_starts.get(r35_key, [])),
        "norepinephrine": any(a[0] == "norepinephrine"
                             for a in r33_starts.get(r35_key, [])),
        "rmse": float((r34_best_pred[r35_j]-r33_target[r35_j])
                      .square().mean().sqrt())
    })
r35_diag = pd.DataFrame(r35_diag)
print("Mismatch by bundle composition:")
print(r35_diag.groupby(["n_start", "n_end", "insulin", "norepinephrine"])
      .rmse.agg(["count", "mean"]).to_string())

r35_atoms_base, r35_owner = r33_action_embedding("none", (), True, "left")
r35_start_mask = []
for r35_i in r33_sample:
    r35_key = (int(r33_ids[r35_i]), round(float(r33_hours[r35_i]), 4))
    r35_start_mask.extend([True]*len(r33_starts.get(r35_key, [])))
    r35_start_mask.extend([False]*len(r33_ends.get(r35_key, [])))
r35_start_mask = torch.tensor(r35_start_mask)
r35_end_mask = ~r35_start_mask
r35_atom_codes = []
for r35_i in r33_sample:
    r35_key = (int(r33_ids[r35_i]), round(float(r33_hours[r35_i]), 4))
    r35_atom_codes += [a[0] for a in r33_starts.get(r35_key, [])]
    r35_atom_codes += [a[0] for a in r33_ends.get(r35_key, [])]
r35_unit_idx = torch.tensor([4 if c == "insulin" else 5
                            for c in r35_atom_codes])
r35_rate_unit_idx = torch.tensor([2 if c == "insulin" else 3
                                 for c in r35_atom_codes])

def r35_set(atoms, mask, prefix, count_all, empty_zero):
    own_counts = torch.zeros(len(r33_sample))
    own_counts.index_add_(0, r35_owner[mask], torch.ones(int(mask.sum())))
    counts = r33_counts if count_all else own_counts
    sums = torch.zeros(len(r33_sample), 192)
    if bool(mask.any()):
        sums.index_add_(0, r35_owner[mask],
                        r33_mlp(atoms[mask], prefix+".mlp1", "gelu"))
    result = r33_ln(r33_mlp(
        sums/counts.clamp_min(1).sqrt()[:, None]
        + r33_linear(counts.log1p()[:, None], prefix+".size"),
        prefix+".mlp2", "gelu"), prefix+".norm")
    if empty_zero:
        result[own_counts == 0] = 0
    return result

r35_context = r34_contexts[("gelu", "out_plus_flow")]
r35_results = []
r35_best_error = float("inf")
for r35_ekind, r35_end_unit, r35_end_rate_unit, r35_numeric_bias, \
    r35_start_unit in itertools.product(
        [0, 2], [False, True], [False, True], [False, True], [False, True]):
    r35_atoms = r35_atoms_base.clone()
    r35_atoms[r35_end_mask] += (
        r33_weights["embed.ekind.weight"][r35_ekind]
        - r33_weights["embed.ekind.weight"][2])
    if r35_end_unit:
        r35_atoms[r35_end_mask] += r33_weights["embed.categorical.unit.weight"][
            r35_unit_idx[r35_end_mask]]
    if r35_end_rate_unit:
        r35_atoms[r35_end_mask] += r33_weights[
            "embed.categorical.rate_unit.weight"][r35_rate_unit_idx[r35_end_mask]]
    if not r35_numeric_bias:
        r35_atoms[r35_end_mask] -= r33_weights["embed.numeric.bias"]
    if not r35_start_unit:
        r35_atoms[r35_start_mask] -= r33_weights[
            "embed.categorical.unit.weight"][r35_unit_idx[r35_start_mask]]
    for r35_e_all, r35_u_all in itertools.product([False, True], repeat=2):
        r35_e = r35_set(r35_atoms, r35_end_mask, "encode", r35_e_all, False)
        r35_u = r35_set(r35_atoms, r35_start_mask, "encode_action", r35_u_all, True)
        r35_input = torch.cat([r35_context, r35_e, r35_u], 1)
        r35_hidden = F.gelu(r33_linear(r35_input, "z_post.0"))
        r35_pred = r33_linear(r35_hidden, "z_post.2")[:, :32]
        r35_error = float((r35_pred-r33_target).square().mean().sqrt())
        r35_cfg = (r35_ekind, r35_end_unit, r35_end_rate_unit,
                   r35_numeric_bias, r35_start_unit, r35_e_all, r35_u_all)
        r35_results.append((r35_error, r35_cfg))
        if r35_error < r35_best_error:
            r35_best_error = r35_error
            r35_best_cfg = r35_cfg
            r35_best_input = r35_input.clone()
            r35_best_hidden = r35_hidden.clone()
            r35_best_pred = r35_pred.clone()

r35_results.sort(key=lambda x: x[0])
print("\nBest metadata/pooling candidates:")
for r35_result in r35_results[:12]:
    print(r35_result)
print("Best latent RMSE:", r35_best_error)

with open("/work/notes.md", "a") as r35_file:
    r35_file.write(
        "\n## Run 35\n"
        "END atoms enter the observation encoder; start atoms enter the action "
        "encoder. GELU and nonzero empty observation encoding are strongly "
        "supported by posterior reconstruction. Checked END metadata and "
        "pooling counts.\n"
        f"Best: {r35_results[:3]}\n"
        "Counts: 35 Python runs, 6 forward passes. No edits.\n")
```
