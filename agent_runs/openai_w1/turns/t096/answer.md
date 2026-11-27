Changing the unit IDs reduced posterior error about fivefold, but did not eliminate it. MAP ID 2 is strongly favored; the glucose ID remains ambiguous. I’ll also test how values exactly on bin edges are assigned. A near-zero error for one unit/bin combination would validate the reconstruction; a remaining error would require another explanation.

```python
import numpy as np
import torch

r87_value_dir = r33_weights["embed.numeric.weight"][:, 0]
r87_value_z = (
    (r79_atoms - r86_templates[r86_nearest]) @ r87_value_dir
    / r87_value_dir.square().sum()
)
r87_current_bins = (r86_nearest % 16 + 1).numpy()
r87_labels = np.array(r86_labels, dtype=object)[r86_nearest.numpy()]
r87_values = np.full(len(r79_atoms), np.nan)
r87_bin_delta = torch.zeros_like(r79_atoms)

r87_record_values = r33_records.loc[
    r33_records["event"].isin(sensors), "value"
].dropna().to_numpy()
r87_integer_readings = bool(np.allclose(r87_record_values,
                                       np.rint(r87_record_values)))
print("record readings are integers:", r87_integer_readings)

print("\n=== RECOVERED VALUES AND BIN BOUNDARIES ===")
for r87_code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    r87_mask = r86_is_sensor.numpy() & (r87_labels == r87_code)
    r87_idx = np.flatnonzero(r87_mask)
    r87_meta = r33_numeric[(r87_code, "value")]
    r87_x = (float(r87_meta["median"])
             + float(r87_meta["iqr"]) * r87_value_z[r87_idx].numpy())
    r87_round_error = np.max(np.abs(r87_x - np.rint(r87_x)))
    if r87_integer_readings:
        r87_x = np.rint(r87_x)
    r87_values[r87_idx] = r87_x
    r87_left = 1 + np.searchsorted(r87_meta["fine_bin_edges"], r87_x, side="left")
    r87_right = 1 + np.searchsorted(r87_meta["fine_bin_edges"], r87_x, side="right")
    r87_bin_delta[r87_idx] = (
        r86_bins[torch.tensor(r87_right)]
        - r86_bins[torch.tensor(r87_current_bins[r87_idx])]
    )
    print(r87_code,
          "max inverse-rounding error", float(r87_round_error),
          "current != left", int(np.sum(r87_current_bins[r87_idx] != r87_left)),
          "edge equalities", int(np.sum(r87_left != r87_right)))

r87_local_delta = r87_bin_delta[r86_take]
r87_gdelta = r87_local_delta * r86_g[:, None]
r87_mdelta = r87_local_delta * r86_m[:, None]
r87_results = []

for r87_gside in range(2):
    for r87_mside in range(2):
        r87_side_atoms = (r86_atoms + r87_gside * r87_gdelta
                         + r87_mside * r87_mdelta)
        for r87_gu in range(len(r86_unit)):
            for r87_mu in range(len(r86_unit)):
                r87_trial_atoms = (
                    r87_side_atoms
                    + r86_g[:, None] * (r86_unit[r87_gu] - r86_unit[2])
                    + r86_m[:, None] * (r86_unit[r87_mu] - r86_unit[3])
                )
                r87_e = r86_encode_observations(r87_trial_atoms)
                r87_z = r79_mlp(
                    torch.cat([r86_ctx, r87_e, r86_u], dim=1),
                    r86_zprefix, "gelu"
                )[:, :32]
                r87_rmse = float((r87_z - r86_target).square().mean().sqrt())
                r87_results.append(
                    (r87_rmse, r87_gu, r87_mu, r87_gside, r87_mside)
                )

r87_results.sort()
print("\n=== BEST COMBINATIONS ===")
print("RMSE, glucose unit, MAP unit, glucose bin-right, MAP bin-right")
for r87_result in r87_results[:12]:
    print(r87_result)

r87_best = r87_results[0]
_, r87_gu, r87_mu, r87_gside, r87_mside = r87_best
r87_atoms_corrected = (
    r79_atoms
    + r86_is_g[:, None] * (r86_unit[r87_gu] - r86_unit[2])
    + r86_is_m[:, None] * (r86_unit[r87_mu] - r86_unit[3])
    + r87_bin_delta * (
        r87_gside * r86_is_g[:, None] + r87_mside * r86_is_m[:, None]
    )
)

# Validate on every sensor bundle, not only the grid-test sample.
r87_obs = ~r79_action
r87_counts_full = torch.bincount(
    r79_owner[r87_obs], minlength=r79_n
).float()
r87_hidden_full = r79_mlp(
    r87_atoms_corrected[r87_obs], "encode.mlp1", "gelu"
)
r87_summed_full = torch.zeros((r79_n, 192))
r87_summed_full.index_add_(0, r79_owner[r87_obs], r87_hidden_full)
r87_pooled_full = (
    r87_summed_full / r87_counts_full.clamp_min(1).sqrt()[:, None]
    + r79_lin(torch.log1p(r87_counts_full)[:, None], "encode.size")
)
r87_encoded_full = r79_norm(
    r79_mlp(r87_pooled_full, "encode.mlp2", "gelu"), "encode.norm"
)
r87_e_full = r80_e.clone()
r87_sensor_rows = r86_sensor_counts > 0
r87_e_full[r87_sensor_rows] = r87_encoded_full[r87_sensor_rows]
r87_input_full = r80_input.clone()
r87_input_full[:, 192:384] = r87_e_full
r87_z_full = r79_mlp(r87_input_full, r86_zprefix, "gelu")[:, :32]
r87_row_error = (r87_z_full - r79_target).square().mean(1).sqrt()
print("\nfull-sensor-bundle RMSE quantiles:",
      torch.quantile(r87_row_error[r87_sensor_rows],
                     torch.tensor([0.0, 0.5, 0.9, 1.0])).tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 87\n"
        "Unit IDs alone reduced but did not eliminate sensor posterior error. "
        "Tested unit IDs jointly with bin-edge equality conventions. "
        "Best combinations: " + repr(r87_results[:4]) + ". "
        "Full-sensor row-error quantiles: " +
        repr(torch.quantile(r87_row_error[r87_sensor_rows],
                            torch.tensor([0.0, 0.5, 0.9, 1.0])).tolist()) +
        ". No weights changed. Counts: 87 Python runs, 6 forward passes.\n"
    )
```