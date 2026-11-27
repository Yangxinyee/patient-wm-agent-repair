Single-infusion posteriors now match the saved states to about **0.00002 RMS**. The correct conventions use left-sided bin boundaries and an encoded empty observation bundle—not a zero vector. I’ll validate infusion ends and measurement bundles next. Comparable agreement there would let us fit a dose-deviation feature on correctly reconstructed latent inputs.

```python
def latent_context_at(indices, state_data=latent_probe_state):
    indices = np.asarray(indices, dtype=int)
    old = torch.as_tensor(state_data["x"][indices - 1])
    dt = torch.as_tensor(
        (state_data["hour"][indices] - state_data["hour"][indices - 1]) * 3600,
        dtype=torch.float32
    )[:, None]
    h_flow = old[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt[:, :, None] * torch.exp(wt("core.flow.a"))[None, None, :]
    )
    return old[:, :192] + ln(h_flow.mean(1), "core.out_norm") + mlp(
        torch.log1p(dt) / 10, "gap", "gelu"
    )

def encode_single_debug(atoms, prefix):
    hidden = mlp(atoms, prefix + ".mlp1", "gelu")
    hidden = hidden + TF.linear(
        torch.full((len(atoms), 1), float(np.log(2))),
        wt(prefix + ".size.weight"), wt(prefix + ".size.bias")
    )
    return ln(mlp(hidden, prefix + ".mlp2", "gelu"), prefix + ".norm")

end_records = raw_probe[
    raw_probe["event"].isin(["insulin", "norepinephrine"])
][["stay_id", "hour", "end_hour", "event", "dose", "rate"]].rename(
    columns={"hour": "start_hour", "end_hour": "hour"}
)
single_end = state_frame[
    (state_frame["n_atoms"] == 1) &
    ~state_frame["has_insulin"] & ~state_frame["has_norepinephrine"]
].merge(end_records, on=["stay_id", "hour"], how="inner")
single_end = single_end[single_end["state_index"] > 0]
single_end = single_end.groupby("event", group_keys=False).head(6)
end_debug_best = {}

for drug in ["insulin", "norepinephrine"]:
    rows = single_end[single_end["event"] == drug]
    if rows.empty:
        continue
    idx = rows["state_index"].to_numpy(dtype=int)
    context = latent_context_at(idx)
    truth = torch.as_tensor(latent_probe_state["x"][idx, 960:])
    code = next(c["id"] for c in VOCAB["codes"] if c["name"] == drug)
    best = (float("inf"), None)

    for numeric_mode in ["absent", "original", "zero_both", "rate_original", "rate_zero"]:
        nums = np.zeros((len(rows), 6), dtype=np.float32)
        atom_bins = {}
        for attr, col, has_col in [("dose", 2, 3), ("rate", 4, 5)]:
            present = numeric_mode in ["original", "zero_both"] or (
                attr == "rate" and numeric_mode in ["rate_original", "rate_zero"]
            )
            values = rows[attr].to_numpy() if numeric_mode in ["original", "rate_original"] else np.zeros(len(rows))
            spec = numeric_specs[(drug, attr)]
            if present:
                nums[:, col] = (values - spec["median"]) / spec["iqr"]
                nums[:, has_col] = 1
                atom_bins[attr] = np.searchsorted(spec["fine_bin_edges"], values, side="left") + 1
            else:
                atom_bins[attr] = np.zeros(len(rows), dtype=int)

        atoms = (
            wt("embed.code.weight")[code] + wt("embed.kind.weight")[2] +
            wt("embed.ekind.weight")[2] +
            TF.linear(torch.as_tensor(nums), wt("embed.numeric.weight"), wt("embed.numeric.bias")) +
            wt("embed.bins.value.weight")[0] +
            wt("embed.bins.dose.weight")[torch.as_tensor(atom_bins["dose"])] +
            wt("embed.bins.rate.weight")[torch.as_tensor(atom_bins["rate"])]
        )
        grid = (atoms[:, None, :] + cat_offsets[None, :, :]).reshape(-1, 192)
        for placement in ["action", "observation"]:
            enc = encode_single_debug(grid, "encode_action" if placement == "action" else "encode")
            enc = enc.reshape(len(rows), len(cat_combinations), 192)
            ctx_grid = context[:, None, :].expand_as(enc)
            e_grid = empty_e[:, None, :].expand_as(enc) if placement == "action" else enc
            u_grid = enc if placement == "action" else torch.zeros_like(enc)
            post = mlp(torch.cat([ctx_grid, e_grid, u_grid], -1).reshape(-1, 576), "z_post", "gelu")
            pred_z = post[:, :32].reshape(len(rows), len(cat_combinations), 32)
            err = ((pred_z - truth[:, None, :]) ** 2).mean((0, 2)).sqrt()
            ci = int(err.argmin())
            result = (float(err[ci]), (numeric_mode, placement, tuple(cat_combinations[ci])))
            if result[0] < best[0]:
                best = result
                end_debug_best[drug] = result
    print("END BEST", drug, best)

measurement_names = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
measurement_groups = {
    key: group for key, group in raw_probe[raw_probe["event"].isin(measurement_names)].groupby(["stay_id", "hour"])
}
measurement_rows = []
for row in state_frame.itertuples():
    group = measurement_groups.get((row.stay_id, row.hour))
    if (group is not None and len(group) == row.n_atoms and row.state_index > 0
            and latent_probe_state["stay_id"][row.state_index - 1] == row.stay_id):
        measurement_rows.append((row.state_index, group))
    if len(measurement_rows) >= 16:
        break

idx = np.array([row[0] for row in measurement_rows])
context = latent_context_at(idx)
truth = torch.as_tensor(latent_probe_state["x"][idx, 960:])
measurement_unit_results = []

for glu_unit, map_unit in itertools.product([1, 2, 5], repeat=2):
    atom_vectors, atom_bundle_indices = [], []
    for bi, (_, group) in enumerate(measurement_rows):
        for row in group.itertuples():
            code = next(c["id"] for c in VOCAB["codes"] if c["name"] == row.event)
            spec = numeric_specs[(row.event, "value")]
            nums = torch.zeros(6)
            nums[0] = (row.value - spec["median"]) / spec["iqr"]
            nums[1] = 1
            value_bin = int(np.searchsorted(spec["fine_bin_edges"], row.value, side="left")) + 1
            unit = glu_unit if row.event.startswith("glucose") else map_unit
            atom_vectors.append(
                wt("embed.code.weight")[code] + wt("embed.kind.weight")[0] +
                wt("embed.ekind.weight")[1] + wt("embed.kq.weight")[0] +
                wt("embed.categorical.unit.weight")[unit] +
                TF.linear(nums, wt("embed.numeric.weight"), wt("embed.numeric.bias")) +
                wt("embed.bins.value.weight")[value_bin] +
                wt("embed.bins.dose.weight")[0] + wt("embed.bins.rate.weight")[0]
            )
            atom_bundle_indices.append(bi)
    atoms = torch.stack(atom_vectors)
    bi = torch.as_tensor(atom_bundle_indices)
    counts = torch.bincount(bi, minlength=len(idx)).float()
    summed = torch.zeros((len(idx), 192)).index_add_(0, bi, mlp(atoms, "encode.mlp1", "gelu"))
    pooled = summed / counts.sqrt()[:, None] + TF.linear(
        torch.log1p(counts)[:, None], wt("encode.size.weight"), wt("encode.size.bias")
    )
    e = ln(mlp(pooled, "encode.mlp2", "gelu"), "encode.norm")
    pred_z = mlp(torch.cat([context, e, torch.zeros_like(e)], -1), "z_post", "gelu")[:, :32]
    error = float(((pred_z - truth) ** 2).mean().sqrt())
    measurement_unit_results.append((error, glu_unit, map_unit))

print("\nMEASUREMENT UNIT TESTS:")
for result in sorted(measurement_unit_results)[:4]:
    print(result)

with notes_path.open("a") as f:
    f.write("\n- Run 46: validated END/measurement conventions before latent repair.\n")
    f.write(f"  - END results: {end_debug_best}.\n")
    f.write(f"  - Best measurement-unit result: {min(measurement_unit_results)}.\n")
```