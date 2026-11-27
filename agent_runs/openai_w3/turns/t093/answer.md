```python
from pathlib import Path
import numpy as np
import torch

print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRETAINED HELPERS AND ARRAYS")
for k in sorted(list(globals())):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    v = globals()[k]
    if callable(v):
        if getattr(v, "__module__", "") == "__main__":
            print(k, "function")
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict) and any(s in k.lower() for s in ("fit", "state", "probe", "weight", "param", "coef")):
        print(k, "dict keys", list(v)[:18])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 83; forwards 7. Restart recovery: the seconds/current-state update reproduces authoritative out with RMSE 1.44e-5 and h with RMSE .00204. Recovering equations/helpers to target the effective prelude/recurrent action path, with all weights original.\n")
```