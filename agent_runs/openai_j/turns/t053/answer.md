The saved bundles include infusion-end atoms, so treating every drug bundle as a fresh dose would be wrong. I’ll start with single-atom infusion bundles, where the encoding is unambiguous, and test the categorical and timing conventions. Near-exact posterior reconstruction would validate this machinery; a substantial residual would stop the proposed latent edit.

```python
import itertools
import torch
import torch.nn.functional as TF

torch.set_num_threads(8)
torch.set_grad_enabled(False)

cat_names = [
    "embed.categorical.unit.weight",
    "embed.categorical.rate_unit.weight",
    "embed.categorical.action.weight",
    "embed.categorical.route.weight",
    "embed.categorical.status.weight",
    "embed.kq.weight",
    "embed.text.weight",
]
print("CATEGORICAL ROW NORMS:")
for name in cat_names:
    print(name, np.round(np.linalg.norm(load_original(name), axis=1), 4).tolist())

state_frame = pd.DataFrame({
    key: latent_probe_state[key]
    for key in ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms")
})
state_frame["state_index"] = np.arange(len(state_frame))
raw_probe = train[train["stay_id"].isin(latent_probe_ids)].copy()
single = state_frame[
    (state_frame["n_atoms"] == 1) &
    (state_frame["has_insulin"] ^ state_frame["has_norepinephrine"])
].merge(
    raw_probe[raw_probe["event"].isin(["insulin", "norepinephrine"])],
    on=["stay_id", "hour"], how="inner"
)
single = single[
    (single["state_index"] > 0) &
    (latent_probe_state["stay_id"][single["state_index"].to_numpy() - 1]
     == single["stay_id"].to_numpy())
]
single = single.groupby("event", group_keys=False).head(8).reset_index(drop=True)
print("\nSINGLE-ACTION VALIDATION ROWS:")
print(single[["stay_id", "hour", "event", "rate", "dose"]].to_string(index=False))

cat_combinations = np.array(list(itertools.product(
    *[range(SHAPES[name][0]) for name in cat_names]
)), dtype=int)
cat_offsets = sum(
    wt(name)[torch.as_tensor(cat_combinations[:, j])]
    for j, name in enumerate(cat_names)
)
numeric_specs = {
    (entry["code"], entry["attribute"]): entry
    for entry in VOCAB["numeric"]
}

empty_e = ln(
    mlp(
        TF.linear(torch.zeros((1, 1)), wt("encode.size.weight"), wt("encode.size.bias")),
        "encode.mlp2", "gelu"
    ), "encode.norm"
)
posterior_debug_results = []
posterior_debug_best = {}

for drug in ["insulin", "norepinephrine"]:
    rows = single[single["event"] == drug]
    if len(rows) == 0:
        continue
    idx = rows["state_index"].to_numpy(dtype=int)
    old = torch.as_tensor(latent_probe_state["x"][idx - 1])
    dt_s = torch.as_tensor(
        (rows["hour"].to_numpy() - latent_probe_state["hour"][idx - 1]) * 3600,
        dtype=torch.float32
    )[:, None]
    flowed = old[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt_s[:, :, None] * torch.exp(wt("core.flow.a"))[None, None, :]
    )
    context_base = old[:, :192] + ln(flowed.mean(1), "core.out_norm")
    truth_z = torch.as_tensor(latent_probe_state["x"][idx, 960:])
    code_id = next(c["id"] for c in VOCAB["codes"] if c["name"] == drug)

    numbers = np.zeros((len(rows), 6), dtype=np.float32)
    for attr, col, has_col in [("dose", 2, 3), ("rate", 4, 5)]:
        spec = numeric_specs[(drug, attr)]
        numbers[:, col] = (rows[attr].to_numpy() - spec["median"]) / spec["iqr"]
        numbers[:, has_col] = 1
    base_atoms = (
        wt("embed.code.weight")[code_id] +
        wt("embed.kind.weight")[0] +
        wt("embed.ekind.weight")[2] +
        TF.linear(torch.as_tensor(numbers), wt("embed.numeric.weight"), wt("embed.numeric.bias"))
    )

    best = (float("inf"), None)
    for side in ["left", "right"]:
        atoms = base_atoms.clone()
        for attr in ["dose", "rate"]:
            bins = np.searchsorted(
                numeric_specs[(drug, attr)]["fine_bin_edges"],
                rows[attr].to_numpy(), side=side
            ) + 1
            atoms += wt(f"embed.bins.{attr}.weight")[torch.as_tensor(bins)]
        atoms += wt("embed.bins.value.weight")[0]
        atom_grid = atoms[:, None, :] + cat_offsets[None, :, :]
        flat = atom_grid.reshape(-1, 192)
        encoded = mlp(flat, "encode_action.mlp1", "gelu")
        encoded += TF.linear(
            torch.full((len(flat), 1), float(np.log(2)), dtype=torch.float32),
            wt("encode_action.size.weight"), wt("encode_action.size.bias")
        )
        u_grid = ln(
            mlp(encoded, "encode_action.mlp2", "gelu"),
            "encode_action.norm"
        ).reshape(len(rows), len(cat_combinations), 192)

        for gap_scale in [10.0, 1.0]:
            context = context_base + mlp(torch.log1p(dt_s) / gap_scale, "gap", "gelu")
            for empty_label, e in [("zero", torch.zeros_like(context)),
                                   ("encoded_empty", empty_e.expand_as(context))]:
                post_input = torch.cat([
                    context[:, None, :].expand_as(u_grid),
                    e[:, None, :].expand_as(u_grid),
                    u_grid
                ], dim=-1)
                post = mlp(post_input.reshape(-1, 576), "z_post", "gelu")
                post_z = post[:, :32].reshape(len(rows), len(cat_combinations), 32)
                errors = ((post_z - truth_z[:, None, :]) ** 2).mean((0, 2)).sqrt()
                ci = int(errors.argmin())
                error = float(errors[ci])
                config = (side, gap_scale, empty_label, tuple(cat_combinations[ci]))
                posterior_debug_results.append((drug, error, config))
                if error < best[0]:
                    best = (error, config)
                    posterior_debug_best[drug] = {
                        "rows": rows.copy(), "context": context.clone(),
                        "e": e.clone(), "u": u_grid[:, ci].clone(),
                        "predicted_z": post_z[:, ci].clone(),
                        "truth_z": truth_z.clone(), "config": config,
                    }
    print("\nBEST", drug, "posterior RMSE:", best[0], "configuration:", best[1])

print("\nTOP RECONSTRUCTIONS:")
for drug, error, config in sorted(posterior_debug_results, key=lambda r: r[1])[:8]:
    print(drug, round(error, 7), config)

with notes_path.open("a") as f:
    f.write("\n- Run 45: single-action posterior reconstruction convention test; no edits.\n")
    for drug, result in posterior_debug_best.items():
        error = float(((result["predicted_z"] - result["truth_z"]) ** 2).mean().sqrt())
        f.write(f"  - {drug}: posterior RMSE {error:.7g}, configuration {result['config']}.\n")
```