The retained helper uses categorical code 0 for every reading, whereas the validated lab-glucose convention is 5. I’ll correct that one convention and test all noninitial bundles. Near-zero posterior errors across compositions would confirm the reconstruction is usable; remaining systematic errors would require further checking.

```python
import numpy as np
import torch
from pathlib import Path

r23_base_embed = embed_consistent_audit

def embed_consistent_audit(row, kind, end_numeric="absent",
                           end_categories="absent", end_kq=0):
    result = r23_base_embed(row, kind, end_numeric, end_categories, end_kq)
    if row.event == "glucose_lab" and kind == 0:
        result = result + W0["embed.kq.weight"][5] - W0["embed.kq.weight"][0]
    return result

r23_indices = np.arange(len(state_keys_audit))
all_e_audit, all_u_audit = bundle_encodings_audit(
    r23_indices, "world", "absent", "absent", 0
)
post_inputs_audit = torch.cat(
    [contexts_audit, all_e_audit, all_u_audit], dim=1
)
post_hidden_audit = act0(lin0(post_inputs_audit, "core.z_post.0"), "gelu")
all_post_means_audit = lin0(post_hidden_audit, "core.z_post.2")[:, :32]
r23_target = torch.as_tensor(sx[:, -32:], dtype=torch.float32)
posterior_errors_audit = (
    (all_post_means_audit - r23_target).square().mean(dim=1).sqrt().numpy()
)

print("=== FULL NONINITIAL POSTERIOR CHECK ===")
r23_errors = posterior_errors_audit[valid_previous_audit]
print("bundles:", len(r23_errors))
print("RMS:", float(np.sqrt(np.mean(r23_errors ** 2))))
print("median / p95 / max:", np.quantile(r23_errors, [0.5, 0.95, 1.0]))

r23_by_composition = {}
for r23_i in valid_previous_audit:
    r23_composition = tuple(sorted(
        str(r23_row.event)
        for r23_row in bundle_records_audit[state_keys_audit[r23_i]]
    ))
    r23_by_composition.setdefault(r23_composition, []).append(
        float(posterior_errors_audit[r23_i])
    )
for r23_composition, r23_vals in sorted(
    r23_by_composition.items(), key=lambda item: -len(item[1])
):
    print(r23_composition, "n", len(r23_vals),
          "RMS", float(np.sqrt(np.mean(np.square(r23_vals)))))

print("\n=== SMALL RECORD SAMPLE ===")
print(record_rows_audit.head(8).to_string(index=False))

print("\n=== RETAINED DYNAMIC ESTIMATES ===")
for r23_code, r23_est in dynamic_ir.items():
    print(r23_code, repr(r23_est)[:900])
print("state metadata keys:", list(train_states0))
print("panel index:", P.index.names, "columns:", list(P.columns))
print("history features:", feature_names)

with Path("/work/notes.md").open("a") as r23_file:
    r23_file.write(
        "\n## Run 23: lab-specific encoding correction\n"
        "- Corrected glucose_lab kq from 0 to 5 in the reconstruction only; "
        "model weights remain unchanged.\n"
        f"- Full noninitial posterior RMS: "
        f"{float(np.sqrt(np.mean(r23_errors ** 2))):.9g}; "
        f"maximum bundle RMS: {float(np.max(r23_errors)):.9g}.\n"
        "- Python runs: 23; forward passes: 4; no weights edited.\n"
    )
```