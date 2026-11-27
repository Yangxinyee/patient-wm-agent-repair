The original model’s immediate insulin response is much weaker than the record-based estimate, while its norepinephrine response is inconsistent between halving and doubling. The small feature changes suggest a targeted weight correction may be possible, but they do not establish that factual accuracy will survive. I’ll recover the exact head and perturbation objects before constructing a correction. Independent factual and rollout checks—not the fitting equations alone—will determine whether it works.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
_items = list(globals().items())
for _name, _obj in _items:
    if inspect.isfunction(_obj) and getattr(_obj, "__module__", "") == "__main__":
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            print(_name)

print("\n=== RECENT RETAINED OBJECTS ===")
for _name, _obj in _items[-100:]:
    if _name.startswith("_") or inspect.isfunction(_obj):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", len(_obj), "keys:", list(_obj)[:10])
    elif hasattr(_obj, "columns") and hasattr(_obj, "shape"):
        print(_name, "DataFrame", _obj.shape, list(_obj.columns)[:14])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj))
    elif isinstance(_obj, (str, int, float, bool)) and len(str(_obj)) < 160:
        print(_name, repr(_obj))

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 75 (restart)\n"
        "Run-74 isolated next-reading sensitivities are weak for insulin and "
        "nonmonotonic for NE despite exact state reconstruction. Recovering "
        "head/perturbation objects before fitting a factual-regularized repair. "
        "No weights edited; rollout targets still require record-based validation.\n"
    )
```
