The atom helper assumes glucose unit ID 2 and MAP unit ID 3; those assignments have not been verified. I’ll test all unit-ID pairs against the exported posterior states, changing no weights. A near-exact match for one pair would identify the encoding error. If every pair leaves a substantial mismatch, the unit hypothesis is insufficient.

```python
import numpy as np
import torch

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 86\n"
        "Testing measurement categorical unit IDs against aligned exported "
        "posteriors. This changes only local reconstruction arrays, not weights. "
        "Counts: 86 Python runs, 6 forward passes.\n"
    )

r86_zprefix = next(
    n[:-len(".0.weight")] for n in NAMES
    if "z_post" in n and n.endswith(".0.weight")
)
r86_unit = r33_weights["embed.categorical.unit.weight"]
r86_bins = r33_weights["embed.bins.value.weight"]
r86_direction = r33_weights["embed.numeric.weight"][:, 0]
r86_direction = r86_direction / torch.linalg.vector_norm(r86_direction)

# Identify measurement atoms algebraically, allowing any value and fine bin.
r86_templates, r86_labels = [], []
for r86_code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    r86_meta = r33_numeric[(r86_code, "value")]
    r86_med = float(r86_meta["median"])
    r86_b = 1 + np.searchsorted(r86_meta["fine_bin_edges"], r86_med, side="left")
    r86_base = r36_atom(r86_code, value=r86_med) - r86_bins[r86_b]
    for r86_bin in range(1, 17):
        r86_templates.append(r86_base + r86_bins[r86_bin])
        r86_labels.append(r86_code)
r86_templates = torch.stack(r86_templates)
r86_tp = r86_templates - (r86_templates @ r86_direction)[:, None] * r86_direction
r86_ap = r79_atoms - (r79_atoms @ r86_direction)[:, None] * r86_direction
r86_dist = (
    (r86_ap * r86_ap).sum(1)[:, None]
    + (r86_tp * r86_tp).sum(1)[None, :]
    - 2 * r86_ap @ r86_tp.T
)
r86_nearest = r86_dist.argmin(1)
r86_residual = torch.linalg.vector_norm(r86_ap - r86_tp[r86_nearest], dim=1)
r86_is_sensor = (r86_residual < 1e-3) & ~r79_action
r86_is_g = torch.tensor(
    [r86_labels[int(i)].startswith("glucose") for i in r86_nearest]
) & r86_is_sensor
r86_is_m = r86_is_sensor & ~r86_is_g

print("=== ATOM IDENTIFICATION ===")
for r86_code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    r86_mask = torch.tensor(
        [r86_labels[int(i)] == r86_code for i in r86_nearest]
    ) & r86_is_sensor
    print(r86_code, int(r86_mask.sum()))
print("identified-atom residual quantiles:",
      torch.quantile(r86_residual[r86_is_sensor],
                     torch.tensor([0.0, 0.5, 1.0])).tolist())

r86_sensor_counts = torch.zeros(r79_n)
r86_sensor_counts.index_add_(0, r79_owner, r86_is_sensor.float())
r86_all_rows = torch.where(r86_sensor_counts > 0)[0]
r86_rng = np.random.default_rng(8601)
r86_rows = torch.tensor(
    r86_rng.choice(r86_all_rows.numpy(), min(384, len(r86_all_rows)), replace=False)
)
r86_row_lookup = torch.full((r79_n,), -1, dtype=torch.long)
r86_row_lookup[r86_rows] = torch.arange(len(r86_rows))
r86_local_owner = r86_row_lookup[r79_owner]
r86_take = (r86_local_owner >= 0) & ~r79_action
r86_owner = r86_local_owner[r86_take]
r86_atoms = r79_atoms[r86_take].clone()
r86_g = r86_is_g[r86_take]
r86_m = r86_is_m[r86_take]
r86_counts = torch.bincount(r86_owner, minlength=len(r86_rows)).float()

def r86_encode_observations(atoms):
    hidden = r79_mlp(atoms, "encode.mlp1", "gelu")
    summed = torch.zeros((len(r86_rows), hidden.shape[1]))
    summed.index_add_(0, r86_owner, hidden)
    pooled = summed / r86_counts.clamp_min(1).sqrt()[:, None]
    pooled = pooled + r79_lin(torch.log1p(r86_counts)[:, None], "encode.size")
    return r79_norm(r79_mlp(pooled, "encode.mlp2", "gelu"), "encode.norm")

r86_e0 = r86_encode_observations(r86_atoms)
print("\n=== BASELINE RECONSTRUCTION CHECK ===")
print("e max error versus cached reconstruction:",
      float((r86_e0 - r80_e[r86_rows]).abs().max()))
r86_ctx = r80_input[r86_rows, :192]
r86_u = r80_input[r86_rows, 384:]
r86_target = r79_target[r86_rows]
r86_results = []
for r86_gu in range(len(r86_unit)):
    for r86_mu in range(len(r86_unit)):
        r86_trial_atoms = (
            r86_atoms
            + r86_g[:, None] * (r86_unit[r86_gu] - r86_unit[2])
            + r86_m[:, None] * (r86_unit[r86_mu] - r86_unit[3])
        )
        r86_e = r86_encode_observations(r86_trial_atoms)
        r86_z = r79_mlp(
            torch.cat([r86_ctx, r86_e, r86_u], dim=1),
            r86_zprefix, "gelu"
        )[:, :32]
        r86_error = r86_z - r86_target
        r86_results.append(
            (float(r86_error.square().mean().sqrt()), r86_gu, r86_mu)
        )

r86_results.sort()
print("\n=== BEST UNIT-ID PAIRS: RMSE, GLUCOSE ID, MAP ID ===")
for r86_result in r86_results[:10]:
    print(r86_result)
print("assumed pair (2,3):",
      next(x for x in r86_results if x[1:] == (2, 3)))

with open("/work/notes.md", "a") as f:
    f.write("Unit-grid results: " + repr(r86_results[:6]) + "\n")
```