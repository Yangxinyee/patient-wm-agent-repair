The unit correction ran, but verification stopped on an incorrect tensor path. That result neither confirms nor refutes the correction. I’ll locate the posterior layers by their actual names and shapes, then compare reconstructed means with the saved GPU states.

```python
import numpy as np
import torch
from pathlib import Path

print("=== POSTERIOR TENSOR NAMES ===")
for r25_name in NAMES:
    if "post" in r25_name:
        print(r25_name, SHAPES[r25_name])

r25_hidden_names = [
    name for name in NAMES
    if name.endswith(".weight") and tuple(SHAPES[name]) == (192, 576)
]
r25_output_names = [
    name for name in NAMES
    if name.endswith(".weight")
    and tuple(SHAPES[name]) == (64, 192)
    and "post" in name
]
assert len(r25_hidden_names) == 1, r25_hidden_names
assert len(r25_output_names) == 1, r25_output_names
posterior_first_prefix = r25_hidden_names[0][:-7]
posterior_output_prefix = r25_output_names[0][:-7]
print("selected:", posterior_first_prefix, posterior_output_prefix)

post_inputs_audit = torch.cat(
    [contexts_audit, all_e_audit, all_u_audit], dim=1
)
post_hidden_audit = act0(
    lin0(post_inputs_audit, posterior_first_prefix), "gelu"
)
all_post_means_audit = lin0(
    post_hidden_audit, posterior_output_prefix
)[:, :32]
r25_target = torch.as_tensor(sx[:, -32:], dtype=torch.float32)
posterior_errors_audit = (
    (all_post_means_audit - r25_target).square().mean(dim=1).sqrt().numpy()
)
r25_errors = posterior_errors_audit[valid_previous_audit]

print("\n=== FULL NONINITIAL POSTERIOR CHECK ===")
print("bundles:", len(r25_errors))
print("RMS:", float(np.sqrt(np.mean(r25_errors ** 2))))
print("median / p95 / max:", np.quantile(r25_errors, [0.5, 0.95, 1.0]))

r25_by_composition = {}
for r25_i in valid_previous_audit:
    r25_composition = tuple(sorted(
        str(row.event)
        for row in bundle_records_audit[state_keys_audit[r25_i]]
    ))
    r25_by_composition.setdefault(r25_composition, []).append(
        float(posterior_errors_audit[r25_i])
    )
for r25_composition, r25_vals in sorted(
    r25_by_composition.items(), key=lambda item: -len(item[1])
):
    print(r25_composition, "n", len(r25_vals),
          "RMS", float(np.sqrt(np.mean(np.square(r25_vals)))))

print("\n=== RECORD AND PANEL METADATA ===")
print(record_rows_audit.head(6).to_string(index=False))
print("state keys:", list(train_states0))
print("panel index:", P.index.names, "columns:", list(P.columns))
print("history features:", feature_names)

print("\n=== RETAINED DYNAMIC ESTIMATES ===")
for r25_code, r25_est in dynamic_ir.items():
    print(r25_code, repr(r25_est)[:700])

with Path("/work/notes.md").open("a") as r25_file:
    r25_file.write(
        "\n## Runs 24–25: exact posterior layer paths\n"
        "- Run 24 corrected the lab unit mapping and encoded bundles, but "
        "verification stopped on a guessed tensor path. No weights changed.\n"
        f"- Located posterior layers: {posterior_first_prefix}, "
        f"{posterior_output_prefix}.\n"
        f"- Full noninitial posterior RMS: "
        f"{float(np.sqrt(np.mean(r25_errors ** 2))):.9g}; "
        f"maximum bundle RMS: {float(np.max(r25_errors)):.9g}.\n"
        "- Python runs: 25; forward passes: 4; no weights edited.\n"
    )
```