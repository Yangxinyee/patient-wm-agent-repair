The probe returned all 4,350 bundle states, but their contents have not yet been compared. I’ll restore the original embedding and recover the architecture and saved analysis. Matching state rows are essential; mismatches would invalidate the dose-direction comparison.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
drop_edit("embed.numeric.weight")
print("Current edits:", edited())

probe_npz = np.load("/work/fwd_5/states_edited.npz")
S_probe = {k: probe_npz[k].copy() for k in probe_npz.files}
print("\n=== PROBE ARRAYS ===")
print({k: (v.shape, str(v.dtype)) for k, v in S_probe.items()})

print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== SAVED NOTEBOOK ===")
print(Path("/work/notes.md").read_text())

print("\n=== KERNEL INVENTORY ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_"):
        continue
    if inspect.isfunction(v):
        try:
            print(k, "function", str(inspect.signature(v)))
        except Exception:
            print(k, "function")
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif hasattr(v, "columns") and hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape, list(v.columns)[:35])
    elif isinstance(v, dict) and k not in ("__builtins__",):
        print(k, "dict", len(v), list(v.keys())[:20])
    elif isinstance(v, (list, tuple)) and len(v) < 150:
        print(k, type(v).__name__, len(v), repr(v)[:400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 27: probe collected, original embedding restored\n"
            "Forward 5 returned 4350 bundle states for the same 100 training stays. "
            "Saved as S_probe; removed temporary embed.numeric.weight edit. "
            "Recovering architecture and kernel analysis before constructing a lasting correction.\n")
```