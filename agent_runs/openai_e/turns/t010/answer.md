The architecture confirms that drugs enter through a separate action encoder, but their effects must pass through the latent and recurrent state before reaching the reading heads. That suggests a targeted edit may be possible. Next I’ll inspect the retained causal analyses and action-layer dimensions; stable effects across identification methods would support an edit, while disagreement would call for more record analysis first.

```python
import numpy as np
import pandas as pd
from pathlib import Path

print("=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== PANEL COLUMNS ===")
print(panel.columns.tolist())
print(panel.head(3).to_string(index=False))
print("\n=== RECORD SAMPLE ===")
print(df.head(8).to_string(index=False))

print("\n=== IDENTIFICATION RESULTS ===")
print("RD table:\n", rd_table.to_string(index=False))
print("Dynamic fit:", dyn_fit)
for k, v in dynamic_rd.items():
    print("dynamic_rd", k, str(v)[:850])
print("NE feature names:", feature_names)
print("NE residual dose SD:", float(np.std(rd)))
print("NE residual outcome SD:", float(np.std(ry)))
print("NE DML effect:", NE_DML)
print("Validation IDs:", VALID_IDS[:12])

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("embed.", "encode", "z_", "last_", "query_", "type_emb", "gap.")):
        print(name, SHAPES[name])

print("\n=== VOCAB ===")
for key in ["config", "codes", "normalisation", "normalization",
            "numeric", "value_bin_centres", "rate_bin_centres"]:
    if key in VOCAB:
        print(key, str(VOCAB[key])[:2600])
print("All vocab keys:", list(VOCAB))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 8: action pathway and identification audit\n"
            "Architecture recovered: action-specific encoder feeds posterior, prelude, "
            "and recurrent state; value heads also have same-code last-reading registers. "
            "Inspecting exact dose support, RD results, and layer dimensions before surgery.\n"
            "Python runs: 8. Forward requests: 2.\n")
```