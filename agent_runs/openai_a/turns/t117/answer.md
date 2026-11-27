The earlier record-based insulin estimate is about −1.93 per unit, with persistence about 0.832; its sustained target is much larger than the model’s current response. Before changing weights again, I’ll validate measurement embeddings against saved posterior means. A near-zero reconstruction error would make those embeddings usable for the memory reconstruction; a larger error would require correcting them first.

```python
import itertools
import torch.nn.functional as F

r103_code_map = {v["name"]: v["id"] for v in VOCAB["codes"]}
r103_groups = {
    (int(s), float(h)): g
    for (s, h), g in rm_records.groupby(["stay_id", "hour"], sort=False)
}
r103_indices = []
r103_bundle_rows = []
for r103_i in range(1, len(rm_states["hour"])):
    if rm_states["stay_id"][r103_i] != rm_states["stay_id"][r103_i - 1]:
        continue
    r103_key = (int(rm_states["stay_id"][r103_i]), float(rm_states["hour"][r103_i]))
    r103_group = r103_groups.get(r103_key)
    if r103_group is None:
        continue
    if not set(r103_group["event"]).issubset(
        {"glucose_finger", "glucose_lab", "map_arterial", "map_cuff"}
    ):
        continue
    r103_indices.append(r103_i)
    r103_bundle_rows.append(r103_group)
    if len(r103_indices) == 32:
        break

r103_idx = np.asarray(r103_indices)
r103_prev = torch.tensor(rm_states["x"][r103_idx - 1], dtype=torch.float32)
r103_dt = torch.tensor(
    (rm_states["hour"][r103_idx] - rm_states["hour"][r103_idx - 1]) * 3600,
    dtype=torch.float32
).reshape(-1, 1)
r103_hflow = r103_prev[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -r103_dt[:, :, None] * torch.exp(rm_W["core.flow.a"])[None, None, :]
)
r103_gap_prefix = next(
    n[:-len(".0.weight")] for n in rm_W
    if "gap" in n and n.endswith(".0.weight")
)
r103_post_prefix = next(
    n[:-len(".0.weight")] for n in rm_W
    if "z_post" in n and n.endswith(".0.weight")
)
r103_ctx = (
    r103_prev[:, :192]
    + rm_ln(r103_hflow.mean(1), "core.out_norm")
    + rm_mlp(torch.log(r103_dt.clamp_min(1e-6)) / 10, r103_gap_prefix)
)
r103_target = torch.tensor(rm_states["x"][r103_idx, 960:992])
r103_counts = np.asarray([len(g) for g in r103_bundle_rows])
r103_flat = pd.concat(r103_bundle_rows, ignore_index=True)
r103_numeric = torch.zeros(len(r103_flat), 6)
r103_bins = []
for r103_j, r103_row in enumerate(r103_flat.itertuples(index=False)):
    r103_spec = rm_normspec[(r103_row.event, "value")]
    r103_numeric[r103_j, 0] = float(np.clip(
        (r103_row.value - r103_spec["median"]) / r103_spec["iqr"], -10, 10
    ))
    r103_numeric[r103_j, 1] = 1
    r103_bins.append(1 + np.searchsorted(
        r103_spec["fine_bin_edges"], r103_row.value, side="left"
    ))
r103_codes = torch.tensor(
    [r103_code_map[e] for e in r103_flat["event"]], dtype=torch.long
)
r103_base = (
    rm_W["embed.code.weight"][r103_codes]
    + rm_W["embed.kind.weight"][0]
    + rm_W["embed.kq.weight"][0]
    + rm_W["embed.bins.value.weight"][torch.tensor(r103_bins)]
    + rm_lin(r103_numeric, "embed.numeric")
)
r103_is_glucose = torch.tensor(
    r103_flat["event"].str.startswith("glucose").to_numpy(), dtype=torch.bool
)
r103_offsets = np.r_[0, np.cumsum(r103_counts)]
r103_unit_count = rm_W["embed.categorical.unit.weight"].shape[0]
r103_ekind_count = rm_W["embed.ekind.weight"].shape[0]

r103_results = []
r103_best = None
for r103_ekind, r103_gu, r103_mu in itertools.product(
    range(r103_ekind_count), range(r103_unit_count), range(r103_unit_count)
):
    r103_units = torch.where(r103_is_glucose, r103_gu, r103_mu)
    r103_atoms = (
        r103_base + rm_W["embed.ekind.weight"][r103_ekind]
        + rm_W["embed.categorical.unit.weight"][r103_units]
        + rm_W["embed.categorical.rate_unit.weight"][0]
    )
    r103_e = torch.empty(len(r103_idx), 192)
    for r103_n in np.unique(r103_counts):
        r103_sel = np.flatnonzero(r103_counts == r103_n)
        r103_batch_atoms = torch.stack([
            r103_atoms[r103_offsets[j]:r103_offsets[j + 1]]
            for j in r103_sel
        ])
        r103_e[r103_sel] = rm_set(r103_batch_atoms, "encode", "gelu")
    r103_z = rm_mlp(
        torch.cat([r103_ctx, r103_e, torch.zeros_like(r103_e)], dim=1),
        r103_post_prefix
    )[:, :32]
    r103_error = float(torch.mean((r103_z - r103_target) ** 2).sqrt())
    r103_results.append((r103_error, r103_ekind, r103_gu, r103_mu))
    if r103_best is None or r103_error < r103_best["rmse"]:
        r103_best = {
            "rmse": r103_error,
            "spec": (r103_ekind, r103_gu, r103_mu),
            "e": r103_e.clone(),
            "z": r103_z.clone()
        }

print("Event kinds:", VOCAB["event_kinds"])
print("Gap/posterior prefixes:", r103_gap_prefix, r103_post_prefix)
print("Measurement posterior candidates:")
for r103_result in sorted(r103_results)[:8]:
    print(r103_result)
print("Target posterior SD:", float(r103_target.std()))

with Path("/work/notes.md").open("a") as r103_f:
    r103_f.write(
        "\n## Measurement-embedding validation; Python 103, forward 11/50\n"
        f"- Best posterior RMSE={r103_best['rmse']:.9g}; "
        f"spec (event kind, glucose unit, MAP unit)={r103_best['spec']}.\n"
        "- No weight changes; memory reconstruction depends on this check.\n"
    )
```
