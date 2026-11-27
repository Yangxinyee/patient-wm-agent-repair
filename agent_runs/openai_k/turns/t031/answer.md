The reconstructed posterior now matches saved states closely: RMS error is about \(6\times10^{-5}\). Only the diagnostic grouping print failed. I’ll inspect the core’s exact tensor layout and the retained dynamic estimates next. These are needed to design a dose-residual correction rather than broadly amplify treatment–condition associations.

```python
from pathlib import Path
import numpy as np

print("=== CONFIGURATION ===")
print(VOCAB["config"])

print("\n=== CORE AND LATENT TENSORS ===")
for r26_name in NAMES:
    if r26_name.startswith(("core.", "z_", "gap.", "out_norm.")):
        print(r26_name, SHAPES[r26_name])

print("\n=== RECORD SAMPLE ===")
print(record_rows_audit.head(5).to_string(index=False))
print("sample bundle:", repr(
    bundle_records_audit[state_keys_audit[valid_previous_audit[0]]]
)[:1300])
print("state metadata:", {
    key: (value.shape if isinstance(value, np.ndarray) else type(value).__name__)
    for key, value in train_states0.items()
})

print("\n=== PANEL AND IDENTIFICATION OBJECTS ===")
print("P index:", P.index.names, "columns:", list(P.columns))
print("P first rows:\n", P.head(3).to_string())
print("history features:", feature_names)
print("valid causal hours:", np.unique(hc))
print("dose distributions:", np.quantile(Tc, [0, .25, .5, .75, .95, .99, 1], axis=0))
for r26_code in measure_codes:
    print("DML", r26_code, dml_results[r26_code])
for r26_code, r26_value in dynamic_ir.items():
    print("dynamic_ir", r26_code, repr(r26_value)[:1400])
for r26_code, r26_value in dynamic_moments.items():
    print("dynamic_moments", r26_code, repr(r26_value)[:1700])
print("dynamic residual keys:", list(dynamic_outcome_residuals))
print("future-dose residual shapes:", {
    key: value.shape for key, value in future_rate_residuals.items()
})
print("cluster dynamic contents:", {
    key: (list(value) if isinstance(value, dict) else type(value).__name__)
    for key, value in cluster_dynamic.items()
})

with Path("/work/notes.md").open("a") as r26_file:
    r26_file.write(
        "\n## Runs 23–26: posterior validation and edit preparation\n"
        "- Lab glucose requires unit code 5, not kq 5. The attempted kq "
        "correction failed without changing model weights.\n"
        "- Posterior layers are z_post.0 and z_post.2, not core.z_post.*.\n"
        "- Full noninitial posterior RMS after correcting the unit is "
        "6.052e-5 (median 9.70e-7; p95 2.225e-5; max 7.016e-4).\n"
        "- Run 25's composition-print failure occurred after successful "
        "reconstruction. Inspecting core layout and dynamic causal objects next.\n"
        "- Python runs: 26; forward passes: 4; no weights edited.\n"
    )
```