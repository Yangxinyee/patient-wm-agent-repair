The weakest head correction nearly preserves factual fit, but its estimated drug responses remain much too small—especially for norepinephrine. Stronger corrections damage factual predictions. I’ll inspect the retained calibration code and architecture before choosing a better pathway; a useful edit must separate dose changes from ordinary factual variation.

```python
from pathlib import Path
import numpy as np
import torch

print(Path("/model/ARCHITECTURE.md").read_text())
print("\n--- relevant tensors ---")
for name in NAMES:
    if any(s in name.lower() for s in ["value", "action", "dose", "prior", "posterior", "head"]):
        print(name, SHAPES[name])

print("\n--- retained objects ---")
for key, obj in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(key, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(key, "dict", len(obj), "keys:", list(obj)[:12])
    elif isinstance(obj, (list, tuple)):
        print(key, type(obj).__name__, len(obj))
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function")
print("\nActive edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 184 — recover head-calibration contracts\n"
            "- Alpha 1 preserves approximate held factual NLL (+.00387), but calibrated effects remain too weak; larger penalties fail factual fit.\n"
            "- Inspecting retained objects and architecture before selecting a better causal pathway. No edits active.\n")
```