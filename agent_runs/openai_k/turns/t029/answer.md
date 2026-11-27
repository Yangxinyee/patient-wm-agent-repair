The correction failed because I misread the candidate tuple: 5 is lab glucose’s **unit** code, not its `kq` code. No weights changed. I’ll restore the original helper, correct the unit mapping, and rerun the full check. Matching lab-containing bundles would confirm this explanation.

```python
import numpy as np
import torch
from pathlib import Path

embed_consistent_audit = r23_base_embed
unit_ids_audit["glucose_lab"] = 5
world_kq_audit = 0

r24_indices = np.arange(len(state_keys_audit))
all_e_audit, all_u_audit = bundle_encodings_audit(
    r24_indices, "world", "absent", "absent", 0
)
post_inputs_audit = torch.cat(
    [contexts_audit, all_e_audit, all_u_audit], dim=1
)
post_hidden_audit = act0(lin0(post_inputs_audit, "core.z_post.0"), "gelu")
all_post_means_audit = lin0(post_hidden_audit, "core.z_post.2")[:, :32]
r24_target = torch.as_tensor(sx[:, -32:], dtype=torch.float32)
posterior_errors_audit = (
    (all_post_means_audit - r24_target).square().mean(dim=1).sqrt().numpy()
)
r24_errors = posterior_errors_audit[valid_previous_audit]

print("=== FULL NONINITIAL POSTERIOR CHECK ===")
print("bundles:", len(r24_errors))
print("RMS:", float(np.sqrt(np.mean(r24_errors ** 2))))
print("median / p95 / max:", np.quantile(r24_errors, [0.5, 0.95, 1.0]))

r24_by_composition = {}
for r24_i in valid_previous_audit:
    r24_composition = tuple(sorted(
        str(r24_row.event)
        for r24_row in bundle_records_audit[state_keys_audit[r24_i]]
    ))
    r24_by_composition.setdefault(r24_composition, []).append(
        float(posterior_errors_audit[r24_i])
    )
for r24_composition, r24_vals in sorted(
    r24_by_composition.items(), key=lambda item: -len(item[1])
):
    print(r24_composition, "n", len(r24_vals),
          "RMS", float(np.sqrt(np.mean(np.square(r24_vals)))))

print("\n=== RECORD AND STATE METADATA ===")
print(record_rows_audit.head(8).to_string(index=False))
print("state keys:", list(train_states0))
print("panel index:", P.index.names, "columns:", list(P.columns))
print("history features:", feature_names)

print("\n=== RETAINED DYNAMIC ESTIMATES ===")
for r24_code, r24_est in dynamic_ir.items():
    print(r24_code, repr(r24_est)[:900])

with Path("/work/notes.md").open("a") as r24_file:
    r24_file.write(
        "\n## Runs 23–24: corrected interpretation of sensor convention\n"
        "- Run 23 failed before testing: candidate value 5 denotes lab unit, "
        "not kq. Corrected unit_ids_audit['glucose_lab']=5; kq remains 0.\n"
        "- Earlier notebook references to lab kq=5 were mistaken.\n"
        f"- Full noninitial posterior RMS: "
        f"{float(np.sqrt(np.mean(r24_errors ** 2))):.9g}; "
        f"maximum bundle RMS: {float(np.max(r24_errors)):.9g}.\n"
        "- Python runs: 24; forward passes: 4; no weights edited.\n"
    )
```