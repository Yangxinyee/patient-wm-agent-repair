The candidate is backed up. The recovered notes show why raw dose–outcome correlations are unsuitable here: adjustment even reverses their signs. I’ll recover the remaining causal checks and the exact correction helpers. Stable history-adjusted effects would support the targets; instability would argue against simply amplifying this edit.

```python
from pathlib import Path
import numpy as np
import torch

r108_notes = Path("/work/notes.md").read_text()
r108_start = r108_notes.find("## Run 4")
print("=== CAUSAL CHECKS AND RESPONSE KERNEL ===")
print(r108_notes[r108_start:r108_start + 4600])

print("=== PYTHON HELPERS ===")
print(sorted(n for n, v in list(globals().items())
             if hasattr(v, "__code__") and not n.startswith("_")))

print("=== RECENT FIT/CANDIDATE VARIABLES ===")
r108_lines = []
for n, v in sorted(list(globals().items())):
    if any(n.startswith(p) for p in
           ["r99", "r100", "r101", "r102", "r103", "r104", "r105", "r106"]):
        if isinstance(v, dict):
            desc = "dict keys=" + str(list(v.keys())[:22])
        elif hasattr(v, "shape"):
            desc = type(v).__name__ + " shape=" + str(tuple(v.shape))
        elif isinstance(v, (str, int, float, list, tuple)):
            desc = str(v)[:170]
        elif callable(v):
            desc = "callable"
        else:
            desc = type(v).__name__
        r108_lines.append(n + ": " + desc)
print("\n".join(r108_lines)[:3400])

print("=== CURRENT WEIGHT DIFFERENCES ===")
for n in edited():
    t = torch.as_tensor(load(n))
    o = torch.as_tensor(load_original(n))
    print(n, tuple(t.shape),
          "L2", float(torch.linalg.vector_norm(t-o)),
          "max", float((t-o).abs().max()))
```