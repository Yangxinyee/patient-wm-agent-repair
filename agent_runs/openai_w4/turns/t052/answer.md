The fixtures now provide everything needed to change a dose while holding the preceding history fixed. I’ll measure how much those changes move the action encoding and posterior, and inspect the retained lag estimates. Weak posterior sensitivity would support an input-to-state bottleneck, though only a rollout test can show whether repairing it improves the readouts.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)

def singleton_action_repair(atom_vectors):
    size_input = torch.full(
        (*atom_vectors.shape[:-1], 1), float(np.log(2)),
        dtype=torch.float32
    )
    pooled = (
        rp_mlp(atom_vectors, "encode_action.mlp1")
        + rp_linear(size_input, "encode_action.size")
    )
    return rp_norm(
        rp_mlp(pooled, "encode_action.mlp2"), "encode_action.norm"
    )

controlled_action_repair = {}
print("=== CONTROLLED DOSE PERTURBATIONS ===")
for r44_drug in ("insulin", "norepinephrine"):
    r44_idx = np.flatnonzero(
        action_fixture_repair["event"].to_numpy() == r44_drug
    )
    r44_rec = action_fixture_repair.iloc[r44_idx]
    r44_original_atoms = action_atom_vectors_repair[r44_drug]
    r44_ctx = action_ctx_repair[r44_idx]
    r44_e = empty_e_repair.expand(len(r44_idx), -1)
    r44_results = {}

    for r44_multiplier in (1.0, 0.5, 2.0):
        r44_delta_numeric = np.zeros((len(r44_idx), 6), np.float32)
        r44_delta_bins = torch.zeros_like(r44_original_atoms)
        for r44_attr, r44_col in (("dose", 2), ("rate", 4)):
            r44_spec = numeric_specs_repair[(r44_drug, r44_attr)]
            r44_old = r44_rec[r44_attr].to_numpy(dtype=float)
            r44_new = r44_old * r44_multiplier
            r44_old_z = np.clip(
                (r44_old - r44_spec["median"]) / r44_spec["iqr"], -10, 10
            )
            r44_new_z = np.clip(
                (r44_new - r44_spec["median"]) / r44_spec["iqr"], -10, 10
            )
            r44_delta_numeric[:, r44_col] = r44_new_z - r44_old_z
            r44_old_bin = 1 + np.searchsorted(
                r44_spec["fine_bin_edges"], r44_old, side="left"
            )
            r44_new_bin = 1 + np.searchsorted(
                r44_spec["fine_bin_edges"], r44_new, side="left"
            )
            r44_bin_table = rW[f"embed.bins.{r44_attr}.weight"]
            r44_delta_bins += (
                r44_bin_table[r44_new_bin] - r44_bin_table[r44_old_bin]
            )

        r44_atoms = (
            r44_original_atoms
            + F.linear(torch.from_numpy(r44_delta_numeric),
                       rW["embed.numeric.weight"])
            + r44_delta_bins
        )
        r44_u = singleton_action_repair(r44_atoms)
        r44_post_input = torch.cat([r44_ctx, r44_e, r44_u], dim=-1)
        r44_hidden = F.gelu(rp_linear(r44_post_input, "z_post.0"))
        r44_z = rp_linear(r44_hidden, "z_post.2")[..., :32]
        r44_results[r44_multiplier] = {
            "atoms": r44_atoms, "u": r44_u,
            "hidden": r44_hidden, "z": r44_z
        }

    r44_baseline = r44_results[1.0]
    r44_rmse = torch.mean(
        (r44_baseline["z"] - posterior_target_repair[r44_idx]) ** 2
    ).sqrt().item()
    print(r44_drug, "baseline posterior RMSE:", round(r44_rmse, 8))
    for r44_multiplier in (0.5, 2.0):
        r44_changed = r44_results[r44_multiplier]
        print(
            " multiplier", r44_multiplier,
            "u-change RMS", round(torch.mean(
                (r44_changed["u"] - r44_baseline["u"]) ** 2
            ).sqrt().item(), 6),
            "posterior-change RMS", round(torch.mean(
                (r44_changed["z"] - r44_baseline["z"]) ** 2
            ).sqrt().item(), 6)
        )
    controlled_action_repair[r44_drug] = {
        "indices": r44_idx, "records": r44_rec, "results": r44_results
    }

print("\n=== RETAINED IMPULSE MOMENTS ===")
for r44_drug, r44_table in impulse_tables_repair.items():
    print(r44_drug)
    print(r44_table.round(5).to_string(index=False))
print("provisional kernels:", provisional_kernels)
print("one-exponential fits:", persistence_repair)

print("\n=== RESIDUAL AND MODEL STRUCTURE ===")
for r44_drug in ("insulin", "norepinephrine"):
    print(r44_drug, {
        k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
        for k, v in dml_residuals[r44_drug].items()
    })
    print("models:", type(dml_models[r44_drug]).__name__,
          repr(dml_models[r44_drug])[:350])
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("current edits:", edited())

with open("/work/notes.md", "a") as r44_file:
    r44_file.write(
        "\n## Run 44\n"
        "Computed controlled half/double-dose perturbations through the reconstructed "
        "singleton action encoder and posterior, holding pre-action context fixed. "
        "Retrieved retained longitudinal moment tables and nuisance-model structure. "
        "No edits; component sensitivity does not establish rollout causality.\n"
    )
```
