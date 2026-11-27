The saved edit still has a validated factual fit, but the CPU reconstruction is not accurate enough to support a stronger edit safely. I’ll recover the architecture and causal evidence before proceeding. A close match to GPU outputs would support synthetic-dose tests; a continued mismatch would rule them out.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK ===")
print(notes_text[:18000])
print("=== RETAINED OBJECTS ===")
keys = sorted(k for k in globals() if not k.startswith("_") and any(
    s in k.lower() for s in
    ("graft", "gate", "ridge", "causal", "rd_", "panel", "register", "transition",
     "synth", "tuning_x", "train_x", "step", "checkpoint", "dose", "original_head")
))
for k in keys:
    obj = globals()[k]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(k, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(k, "dict", len(obj), list(obj)[:12])
    elif callable(obj):
        print(k, "function", getattr(getattr(obj, "__code__", None), "co_varnames", ())[:10])
    else:
        print(k, type(obj).__name__, str(obj)[:220])
print("CURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Recovery after Python 156; forwards 11/50\n"
            "- Retained beta=-2 graft has GPU-validated factual delta about -0.00008.\n"
            "- CPU register-offset audit still mismatches GPU; not adequate for an unvalidated stronger edit.\n"
            "- Recovering architecture, causal evidence, and retained construction before further trials.\n")
```
