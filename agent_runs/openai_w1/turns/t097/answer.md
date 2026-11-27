The bin-edge convention does not explain the remaining error; the documented “left” convention fits better. Many reading bundles now reconstruct almost exactly, but a substantial tail remains. I’ll separate that tail by sensor, compare the retained context calculations, and test unit IDs per sensor. Consistently near-zero errors across all four sensors would validate the reconstruction.

```python
import numpy as np
import torch

r88_codes = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
r88_incidence = {}
print("=== SENSOR-STRATIFIED CURRENT ERROR ===")
print("sensors:", sensors, "drugs:", drugs)
for r88_code in r88_codes:
    r88_amask = r86_is_sensor & torch.tensor(r87_labels == r88_code)
    r88_count = torch.zeros(r79_n)
    r88_count.index_add_(0, r79_owner, r88_amask.float())
    r88_incidence[r88_code] = r88_count > 0
    r88_err = r87_row_error[r88_count > 0]
    print(r88_code, len(r88_err),
          torch.quantile(r88_err, torch.tensor([0.5, 0.9, 1.0])).tolist())

r88_local_masks = {
    c: r86_is_sensor[r86_take] & torch.tensor(r87_labels[r86_take.numpy()] == c)
    for c in r88_codes
}

def r88_local_atoms(unit_ids):
    atoms = r86_atoms.clone()
    for code in r88_codes:
        old = 2 if code.startswith("glucose") else 3
        atoms += r88_local_masks[code][:, None] * (
            r86_unit[unit_ids[code]] - r86_unit[old]
        )
    return atoms

def r88_local_score(unit_ids, ctx):
    e = r86_encode_observations(r88_local_atoms(unit_ids))
    z = r79_mlp(torch.cat([ctx, e, r86_u], 1),
                 r86_zprefix, "gelu")[:, :32]
    return float((z - r86_target).square().mean().sqrt()), e

r88_units = {"glucose_finger": 1, "glucose_lab": 1,
             "map_arterial": 2, "map_cuff": 2}
r88_contexts = {"r80-input": r80_input[:, :192]}
for r88_key, r88_val in r74_contexts.items():
    if hasattr(r88_val, "shape") and tuple(r88_val.shape) == (r79_n, 192):
        r88_contexts[r88_key] = torch.as_tensor(r88_val, dtype=torch.float32)

print("\n=== CONTEXT COMPARISON ===")
r88_context_results = []
for r88_key, r88_ctx in r88_contexts.items():
    r88_score, _ = r88_local_score(r88_units, r88_ctx[r86_rows])
    r88_context_results.append((r88_score, r88_key))
    print(r88_key, r88_score,
          "max difference from r80:",
          float((r88_ctx - r80_input[:, :192]).abs().max()))
r88_context_results.sort()
r88_context_key = r88_context_results[0][1]
r88_ctx_full = r88_contexts[r88_context_key]
r88_ctx_local = r88_ctx_full[r86_rows]

print("\n=== PER-SENSOR UNIT TEST ===")
r88_unit_tests = []
for r88_sweep in range(2):
    for r88_code in r88_codes:
        r88_trials = []
        for r88_id in range(len(r86_unit)):
            r88_trial = dict(r88_units)
            r88_trial[r88_code] = r88_id
            r88_score, _ = r88_local_score(r88_trial, r88_ctx_local)
            r88_trials.append((r88_score, r88_id))
        r88_trials.sort()
        r88_units[r88_code] = r88_trials[0][1]
        r88_unit_tests.append((r88_sweep, r88_code, r88_trials))
        print(r88_sweep, r88_code, r88_trials[:3])
print("selected units:", r88_units)

# Independent full-array check of the selected constant field mappings.
r88_atoms_full = r79_atoms.clone()
for r88_code in r88_codes:
    r88_mask = r86_is_sensor & torch.tensor(r87_labels == r88_code)
    r88_old = 2 if r88_code.startswith("glucose") else 3
    r88_atoms_full += r88_mask[:, None] * (
        r86_unit[r88_units[r88_code]] - r86_unit[r88_old]
    )
r88_hidden_full = r79_mlp(r88_atoms_full[r87_obs], "encode.mlp1", "gelu")
r88_sum_full = torch.zeros((r79_n, 192))
r88_sum_full.index_add_(0, r79_owner[r87_obs], r88_hidden_full)
r88_pool_full = (
    r88_sum_full / r87_counts_full.clamp_min(1).sqrt()[:, None]
    + r79_lin(torch.log1p(r87_counts_full)[:, None], "encode.size")
)
r88_encoded_full = r79_norm(
    r79_mlp(r88_pool_full, "encode.mlp2", "gelu"), "encode.norm"
)
r88_e_full = r80_e.clone()
r88_e_full[r87_sensor_rows] = r88_encoded_full[r87_sensor_rows]
r88_z_full = r79_mlp(
    torch.cat([r88_ctx_full, r88_e_full, r80_input[:, 384:]], 1),
    r86_zprefix, "gelu"
)[:, :32]
r88_row_error = (r88_z_full - r79_target).square().mean(1).sqrt()
print("\n=== FULL-ARRAY VALIDATION ===")
for r88_code in r88_codes:
    r88_err = r88_row_error[r88_incidence[r88_code]]
    print(r88_code,
          torch.quantile(r88_err, torch.tensor([0.0, 0.5, 0.9, 1.0])).tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 88\n"
        "Bin-edge alternatives did not resolve residual error. Compared contexts "
        "and tested unit IDs independently by sensor. Context results: " +
        repr(r88_context_results) + "; selected units: " + repr(r88_units) +
        "; sensor median/90th/max errors: " +
        repr({c: torch.quantile(r88_row_error[r88_incidence[c]],
                               torch.tensor([0.5, 0.9, 1.0])).tolist()
              for c in r88_codes}) +
        ". No weights changed. Counts: 88 Python runs, 6 forward passes.\n"
    )
```