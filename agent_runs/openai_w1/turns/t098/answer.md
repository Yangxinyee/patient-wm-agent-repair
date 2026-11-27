All four reading sensors now reconstruct to floating-point precision. The error came from sensor-specific categorical unit IDs, not bin boundaries or context. This validates the reading-state reconstruction, not the causal effects. I’ll check the drug fields the same way and recover the retained edit candidates. Drug reconstruction should also reach near-zero error before I use it to calibrate an edit.

```python
import numpy as np
import torch

print("=== CATEGORICAL UNIT VECTOR NORMS ===")
print(torch.linalg.vector_norm(r86_unit, dim=1).tolist())

# Identify START drug atoms, allowing separate dose and rate bins.
r89_start = r79_action & ~r79_end
r89_start_indices = torch.where(r89_start)[0]
r89_directions = r33_weights["embed.numeric.weight"][:, 2] + \
                 r33_weights["embed.numeric.weight"][:, 4]
r89_direction = r89_directions / torch.linalg.vector_norm(r89_directions)
r89_templates, r89_labels = [], []
r89_dose_bins = r33_weights["embed.bins.dose.weight"]
r89_rate_bins = r33_weights["embed.bins.rate.weight"]
for r89_code in drugs:
    r89_rm = r33_numeric[(r89_code, "rate")]
    r89_dm = r33_numeric[(r89_code, "dose")]
    r89_rb = 1 + np.searchsorted(r89_rm["fine_bin_edges"], r89_rm["median"],
                                side="left")
    r89_db = 1 + np.searchsorted(r89_dm["fine_bin_edges"], r89_dm["median"],
                                side="left")
    r89_base = (
        r36_atom(r89_code, rate=r89_rm["median"], dose=r89_dm["median"])
        - r89_rate_bins[r89_rb] - r89_dose_bins[r89_db]
    )
    for r89_db in range(1, 17):
        for r89_rb in range(1, 17):
            r89_templates.append(r89_base + r89_dose_bins[r89_db]
                                 + r89_rate_bins[r89_rb])
            r89_labels.append(r89_code)
r89_templates = torch.stack(r89_templates)
r89_tp = r89_templates - (r89_templates @ r89_direction)[:, None] * r89_direction
r89_sa = r79_atoms[r89_start_indices]
r89_sp = r89_sa - (r89_sa @ r89_direction)[:, None] * r89_direction
r89_dist = (
    r89_sp.square().sum(1)[:, None] + r89_tp.square().sum(1)[None, :]
    - 2 * r89_sp @ r89_tp.T
)
r89_nearest = r89_dist.argmin(1)
r89_residual = torch.linalg.vector_norm(r89_sp - r89_tp[r89_nearest], dim=1)
r89_is_insulin = torch.zeros(len(r79_atoms), dtype=torch.bool)
r89_is_ne = torch.zeros_like(r89_is_insulin)
r89_start_labels = np.array(r89_labels, dtype=object)[r89_nearest.numpy()]
r89_is_insulin[r89_start_indices] = torch.tensor(r89_start_labels == "insulin")
r89_is_ne[r89_start_indices] = torch.tensor(r89_start_labels == "norepinephrine")
print("\nSTART classification:",
      int(r89_is_insulin.sum()), int(r89_is_ne.sum()),
      "max residual", float(r89_residual.max()))

r89_action_counts = torch.bincount(
    r79_owner[r79_action], minlength=r79_n
).float()
r89_all_rows = torch.where(r89_action_counts > 0)[0]
r89_rng = np.random.default_rng(8901)
r89_rows = torch.tensor(r89_rng.choice(
    r89_all_rows.numpy(), min(512, len(r89_all_rows)), replace=False
))
r89_lookup = torch.full((r79_n,), -1, dtype=torch.long)
r89_lookup[r89_rows] = torch.arange(len(r89_rows))
r89_owner_all = r89_lookup[r79_owner]
r89_take = (r89_owner_all >= 0) & r79_action
r89_owner = r89_owner_all[r89_take]
r89_atoms = r79_atoms[r89_take]
r89_imask = r89_is_insulin[r89_take]
r89_nmask = r89_is_ne[r89_take]
r89_counts = torch.bincount(r89_owner, minlength=len(r89_rows)).float()

def r89_encode_actions(atoms):
    hidden = r79_mlp(atoms, "encode_action.mlp1", "gelu")
    summed = torch.zeros((len(r89_rows), hidden.shape[1]))
    summed.index_add_(0, r89_owner, hidden)
    pooled = summed / r89_counts.sqrt()[:, None]
    pooled += r79_lin(torch.log1p(r89_counts)[:, None], "encode_action.size")
    return r79_norm(r79_mlp(pooled, "encode_action.mlp2", "gelu"),
                     "encode_action.norm")

r89_ctx = r88_ctx_full[r89_rows]
r89_e = r88_e_full[r89_rows]
r89_target = r79_target[r89_rows]
r89_results = []
for r89_iu in range(-1, len(r86_unit)):
    for r89_nu in range(-1, len(r86_unit)):
        r89_trial = r89_atoms.clone()
        if r89_iu >= 0:
            r89_trial += r89_imask[:, None] * r86_unit[r89_iu]
        if r89_nu >= 0:
            r89_trial += r89_nmask[:, None] * r86_unit[r89_nu]
        r89_u = r89_encode_actions(r89_trial)
        r89_z = r79_mlp(torch.cat([r89_ctx, r89_e, r89_u], 1),
                         r86_zprefix, "gelu")[:, :32]
        r89_results.append(
            (float((r89_z - r89_target).square().mean().sqrt()),
             r89_iu, r89_nu)
        )
r89_results.sort()
print("\n=== DRUG DOSE-UNIT TEST: RMSE, INSULIN ID, NE ID ===")
for r89_result in r89_results[:8]:
    print(r89_result)
print("no added unit:", next(x for x in r89_results if x[1:] == (-1, -1)))

print("\n=== RETAINED EDIT CANDIDATES ===")
for r89_name in sorted(list(globals())):
    if r89_name.startswith(("rr50_", "rr51_", "r58_")):
        r89_obj = globals()[r89_name]
        if callable(r89_obj) or isinstance(r89_obj, dict) or hasattr(r89_obj, "shape"):
            print(r89_name, r85_brief(r89_obj))
            if isinstance(r89_obj, dict):
                print(" keys:", [repr(k)[:45] for k in list(r89_obj)[:8]])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 89\n"
        "Reading reconstruction is exact with units finger=0, lab=5, both MAP=2 "
        "and left-edge bins. Tested additional drug dose-unit fields; best results: "
        + repr(r89_results[:5]) +
        ". Recovered retained candidates; no weights changed. "
        "Counts: 89 Python runs, 6 forward passes.\n"
    )
```