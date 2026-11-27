The action owners index only the 1,602 drug bundles, but pooling treated them as indices into all 4,491 bundles. That explains the mismatch. I’ll map them to the full state sequence and check against the validated encoder; numerical agreement would resolve this blocker without changing any weights.

```python
import numpy as np
import torch
from pathlib import Path

owners72 = np.asarray(drug_indices49)[np.asarray(owners65, dtype=int)]
counts72 = np.bincount(owners72, minlength=len(u_valid60))
u72 = encode_set50(
    emb70, owners72, counts72, "encode_action", zero_if_empty=False
)

print("=== CORRECTED FULL-BUNDLE ALIGNMENT ===")
print("owner range:", int(owners72.min()), int(owners72.max()))
print("active bundle count:", int((counts72 > 0).sum()))
print("full encoder RMS:", float((u72-u_valid60).square().mean().sqrt()))
for label72, mask72 in [
    ("empty", counts72 == 0), ("active", counts72 > 0)
]:
    print(label72, float(
        (u72[torch.as_tensor(mask72)] -
         u_valid60[torch.as_tensor(mask72)]).square().mean().sqrt()
    ))

print("\n=== CONTEXT INTERFACE ===")
print(compact_bytecode71(context64, 1900))
print("\n=== TRANSITION INTERFACE ===")
print("locals:", transition64.__code__.co_varnames)
print("dependencies:", transition64.__code__.co_names)
print(compact_bytecode71(transition64, 2400))
print("\n=== DECODER INTERFACE ===")
print("locals:", decode67.__code__.co_varnames)
print("dependencies:", decode67.__code__.co_names)
print(compact_bytecode71(decode67, 2300))

print("\n=== VALUE-HEAD TENSORS ===")
for n72 in NAMES:
    if "attr_heads.value_bin" in n72 or "embed_transition.value_bin" in n72:
        print(n72, SHAPES[n72])

with Path("/work/notes.md").open("a") as f72:
    f72.write(
        "\n## Run 72 / action-owner alignment fix\n"
        "Located run 65–67 mismatch: owners65 addresses the 1602 drug-bundle "
        "subset, not the 4491 full sequence. Remapped through drug_indices49 "
        "and rebuilt full counts. Testing against validated u_valid60. No edits.\n"
    )
```