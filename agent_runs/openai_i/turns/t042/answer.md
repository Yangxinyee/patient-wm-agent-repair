The state export includes infusion endings as well as starts, so a correction must not mistake an ending for another dose. I’ll first reconstruct single-start action bundles by matching their posterior means. Near-zero reconstruction error would validate the encoding; substantial error would refute the proposed input construction before any weights are changed.

```python
import itertools
import time
import torch.nn.functional as F

torch.set_num_threads(8)
torch.set_grad_enabled(False)
repair_weights = {n: torch.as_tensor(load_original(n)).float() for n in NAMES}

def repair_linear(x, name):
    return F.linear(x, repair_weights[name + ".weight"],
                    repair_weights.get(name + ".bias"))

def repair_norm(x, name):
    return F.layer_norm(x, (x.shape[-1],),
                        repair_weights[name + ".weight"],
                        repair_weights[name + ".bias"])

print("Original helper activation names:", mlp_named.__code__.co_names)
repair_activation = F.gelu if "gelu" in mlp_named.__code__.co_names else F.silu

def repair_mlp(x, name):
    return repair_linear(repair_activation(repair_linear(x, name + ".0")),
                         name + ".2")

def repair_atom_base(row, kind=0, include_numbers=True):
    code = code_ids[row["event"]]
    v = (repair_weights["embed.code.weight"][code]
         + repair_weights["embed.kind.weight"][kind]
         + repair_weights["embed.ekind.weight"][2]
         + repair_weights["embed.numeric.bias"]).clone()
    nums = np.zeros(6, dtype=np.float32)
    if include_numbers:
        for attr, offset in (("dose", 2), ("rate", 4)):
            value = float(row[attr])
            if np.isfinite(value):
                spec = numeric_specs[(row["event"], attr)]
                nums[offset] = np.clip(
                    (value - spec["median"]) / spec["iqr"], -10, 10
                )
                nums[offset + 1] = 1
                b = 1 + np.searchsorted(spec["fine_bin_edges"], value, side="left")
                v += repair_weights["embed.bins." + attr + ".weight"][int(b)]
    return v + repair_linear(torch.from_numpy(nums), "embed.numeric") \
             - repair_weights["embed.numeric.bias"]

repair_train100 = train[train.stay_id.isin(np.unique(state_ids))].copy()
repair_lookup = {
    (int(s), round(float(h), 4)): g
    for (s, h), g in repair_train100.groupby(["stay_id", "hour"], sort=False)
}
repair_single_rows = []
for j in range(1, len(state_ids)):
    if state_ids[j] != state_ids[j-1] or state_pack["n_atoms"][j] != 1:
        continue
    if not (state_pack["has_insulin"][j] or state_pack["has_norepinephrine"][j]):
        continue
    group = repair_lookup.get((int(state_ids[j]), round(float(state_hours[j]), 4)))
    if group is not None and len(group) == 1 and group.iloc[0]["event"] in ("insulin", "norepinephrine"):
        repair_single_rows.append((j, group.iloc[0].to_dict()))

print("Single-start bundles:", len(repair_single_rows))
repair_cat_grid = list(itertools.product(
    range(6), range(4), range(2), range(2), range(2), [-1, 0, 1, 2, 3]
))
repair_encoding_scores = []
repair_encoding_best = {}
_start = time.time()

for drug in ("insulin", "norepinephrine"):
    cases = [(j, r) for j, r in repair_single_rows if r["event"] == drug][:24]
    if not cases:
        continue
    js = np.array([j for j, _ in cases])
    previous = torch.from_numpy(state_x[js-1])
    dt = torch.tensor((state_hours[js] - state_hours[js-1]) * 3600,
                      dtype=torch.float32)
    hflow = previous[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt[:, None, None] * torch.exp(repair_weights["core.flow.a"])[None, None, :]
    )
    base_ctx = previous[:, :192] + repair_norm(hflow.mean(1), "core.out_norm")
    gap_options = {
        "log_seconds_div10": torch.log(dt.clamp_min(1e-6)) / 10,
        "log1p_seconds_div10": torch.log1p(dt) / 10,
        "log_hours_div10": torch.log((dt / 3600).clamp_min(1e-6)) / 10,
        "log_seconds": torch.log(dt.clamp_min(1e-6)),
    }
    contexts = {
        k: base_ctx + repair_mlp(v[:, None], "gap")
        for k, v in gap_options.items()
    }
    target = torch.from_numpy(state_x[js, 960:992])
    atom_base = torch.stack([repair_atom_base(r) for _, r in cases])
    best = (np.inf, None, None, None)
    for start in range(0, len(repair_cat_grid), 48):
        configs = repair_cat_grid[start:start+48]
        additives = []
        for unit, rate_unit, action_cat, route, status, kq in configs:
            a = (
                repair_weights["embed.categorical.unit.weight"][unit]
                + repair_weights["embed.categorical.rate_unit.weight"][rate_unit]
                + repair_weights["embed.categorical.action.weight"][action_cat]
                + repair_weights["embed.categorical.route.weight"][route]
                + repair_weights["embed.categorical.status.weight"][status]
            )
            if kq >= 0:
                a = a + repair_weights["embed.kq.weight"][kq]
            additives.append(a)
        atoms = atom_base[None] + torch.stack(additives)[:, None]
        first = repair_mlp(atoms, "encode_action.mlp1")
        pooled = first + repair_linear(
            torch.full((*first.shape[:-1], 1), np.log(2), dtype=torch.float32),
            "encode_action.size"
        )
        actions = repair_norm(repair_mlp(pooled, "encode_action.mlp2"),
                              "encode_action.norm")
        for gap_name, ctx in contexts.items():
            expanded_ctx = ctx[None].expand(len(configs), -1, -1)
            post = repair_mlp(torch.cat(
                [expanded_ctx, torch.zeros_like(actions), actions], -1
            ), "z_post")
            prior = repair_mlp(ctx, "z_prior")[:, :32]
            for mode, prediction in (
                ("direct", post[..., :32]),
                ("prior_plus", post[..., :32] + prior[None]),
            ):
                errors = ((prediction - target[None])**2).mean((1, 2)).sqrt()
                k = int(errors.argmin())
                value = float(errors[k])
                if value < best[0]:
                    best = (value, configs[k], gap_name, mode)
    repair_encoding_best[drug] = best
    print(drug, "cases", len(cases), "best RMS/config/gap/mode:", best)

print("Elapsed seconds:", round(time.time() - _start, 2))
with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Python 35; forward 6 — posterior reconstruction\n"
        "- Infusion endings are separate atoms; current-drug flags distinguish starts.\n"
        "- Searched categorical mappings and gap conventions on single-start bundles, "
        "matching original exposed posterior means, without editing weights.\n"
        f"- Best encoding checks: {repair_encoding_best!r}\n"
    )
```