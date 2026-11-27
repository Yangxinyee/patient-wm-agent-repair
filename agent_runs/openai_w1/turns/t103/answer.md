The earlier work identified a potentially useful approach: carry explicit dose information through a low-variance action subspace, then read it in the posterior. However, that construction predates the corrected unit fields.

I’ll check the exact atom builder and cached arrays so we can rebuild it correctly. Numerical agreement with the exported posterior is the first acceptance test; held-out dose scaling is the next.

```python
import dis
import io
import numpy as np
import torch
from pathlib import Path

r94_notes = Path("/work/notes.md").read_text()
print("=== COMPLETE KEY CANDIDATE NOTES ===")
for r94_section in r94_notes.split("\n## "):
    r94_header = r94_section.split("\n", 1)[0]
    if r94_header.startswith(("Run 50", "Run 51", "Run 58")):
        print("## " + r94_section[:1800])

print("\n=== ATOM BUILDER ===")
r94_buf = io.StringIO()
dis.dis(r36_atom, file=r94_buf)
print(r94_buf.getvalue()[:3900])

print("\n=== CORRECTED ENCODING AND DIRECTION ARRAYS ===")
for r94_name, r94_obj in sorted(list(globals().items())):
    if r94_name.startswith(("r89_", "rr58_", "r88_ctx", "r86_unit")) and not callable(r94_obj):
        if isinstance(r94_obj, (np.ndarray, torch.Tensor, dict, list, tuple)):
            print(r94_name, type(r94_obj).__name__,
                  tuple(r94_obj.shape) if isinstance(r94_obj, (np.ndarray, torch.Tensor)) else len(r94_obj))
            if isinstance(r94_obj, dict):
                print(" keys:", list(r94_obj)[:10])

print("\n=== BUNDLE AND ACTION INDEX EXAMPLES ===")
print("r36_indices type/length:", type(r36_indices).__name__, len(r36_indices))
print("r36_indices first:", list(r36_indices)[:3])
print("action examples:", repair_action_atoms[:4])
print("context shape:", tuple(r36_ctx.shape))

with open("/work/notes.md", "a") as r94_f:
    r94_f.write(
        "\n## Run 94\n"
        "Recovering exact corrected atom construction and retained latent perturbation arrays. "
        "Transparent action-subspace approach must be rebuilt with correct units, not reused unchanged. "
        "No tensors edited. Counts: 94 Python runs, 6 forward passes.\n"
    )
```
