The reconstructed memory ages match almost exactly, and removing empty-memory entries makes the state reconstruction much worse. That rules out those two explanations for the remaining mismatch. I’ll recover the transition and edit helpers, then test whether dose contrasts are reliable despite that mismatch. Stable contrasts would support using them to build an edit; unstable contrasts would rule that approach out.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== VALUE-HEAD TENSORS ===")
for name in NAMES:
    if "attr_heads.value_bin" in name:
        print(name, SHAPES[name])

print("\n=== NOTEBOOK: EARLY FINDINGS ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:12500])

print("\n=== AVAILABLE NOTEBOOK FUNCTIONS ===")
for name, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(name)

print("\n=== RELEVANT VARIABLES ===")
keywords = ("trans", "step", "graft", "rob", "rd_", "causal", "spec",
            "contrast", "cache", "target", "beta", "checkpoint", "state")
for name, obj in list(globals().items()):
    if name.startswith("_") or not any(k in name.lower() for k in keywords):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:16])
    elif isinstance(obj, (str, int, float, bool, tuple)):
        print(name, repr(obj)[:350])
```
