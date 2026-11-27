The local Monte Carlo check showed only a small NLL increase, but the raw and fitted decoder outputs disagreed substantially. That suggests an implementation mismatch, not a validated repair. I’ll recover the architecture, causal evidence, and coefficient transformations before changing weights again. An exact match between fitted predictions and saved-weight predictions would support this diagnosis; disagreement would refute the reconstruction.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print("Notebook characters:", len(notes_text))
print(notes_text[:min(22000, max(0, len(notes_text) - 8000))])

print("\n=== KERNEL VARIABLE INVENTORY ===")
_g = dict(globals())
print(" ".join(sorted(k for k in _g if not k.startswith("_"))))

print("\n=== REPAIR / HEAD / COEFFICIENT SUMMARIES ===")
for k in sorted(_g):
    if k.startswith(("repair_", "head_", "mc_")) or any(s in k.lower() for s in ("decoder", "coef", "scaler")):
        v = _g[k]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape), str(v.dtype))
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v)[:40])
        elif isinstance(v, (int, float, str, bool, tuple, list)):
            print(k, repr(v)[:600])
        elif callable(v):
            print(k, "callable")
        else:
            print(k, type(v).__name__)

print("\n=== ACTIVE EDITS ===")
print(edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 88: reconstruction audit resumed\n"
            "- No active edits. Last local MC candidate dNLL was +0.00070, "
            "but raw-versus-fitted tilt RMS disagreement was 1.304.\n"
            "- Recovering architecture, identification evidence, and feature transformations "
            "before another candidate. A saved-weight equivalence test is required.\n")
```