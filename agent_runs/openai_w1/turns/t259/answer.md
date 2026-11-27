```python
from pathlib import Path
import numpy as np
import torch

rejected230 = {n: load(n).clone() if torch.is_tensor(load(n)) else np.array(load(n), copy=True)
               for n in list(edited())}
for n in list(edited()):
    drop_edit(n)
print("Restored original weights. Saved rejected candidate:", list(rejected230))
print("\nARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())

def brief230(v):
    if torch.is_tensor(v) or isinstance(v, np.ndarray):
        return f"{type(v).__name__} shape={tuple(v.shape)}"
    if isinstance(v, dict):
        return f"dict({len(v)}), keys={list(v)[:12]}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}({len(v)})"
    if callable(v):
        return "callable"
    return f"{type(v).__name__}: {str(v)[:100]}"

print("\nRECENT/CANDIDATE VARIABLES")
for k in sorted(list(globals())):
    if not k.startswith("_") and any(s in k.lower() for s in
        ["relay", "marker", "kernel", "effect", "dose", "numeric", "candidate", "229", "228", "227", "221", "215"]):
        print(k, brief230(globals()[k]))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 230\nForward 21 rejected: factual delta +0.19319 exceeds .005. "
            "Preserved candidate in rejected230 and restored all originals. "
            "Recovering architecture and cached identification before selecting a new route. "
            "Counts: 230 Python runs, 21 forwards.\n")
```