The architecture confirms that drug inputs alter a shared recurrent state, while the value head is a more selective place to edit. The recovered estimates are still associations, so I won’t treat them as causal targets yet. Next I’ll recover the identification results and decoder variables. Agreement across independent record-based estimates, plus an exact coefficient conversion, would justify another candidate.

```python
import pandas as pd
from pathlib import Path
import numpy as np
import torch

_g = dict(globals())
print("=== FUNCTION NAMES ===")
print(" ".join(sorted(k for k, v in _g.items()
                      if not k.startswith("_") and callable(v)
                      and getattr(v, "__module__", "") == "__main__")))

print("\n=== RELEVANT VARIABLES ===")
for k in sorted(_g):
    if k.startswith(("repair_", "head_", "mc_")) or any(s in k.lower() for s in ("decoder", "coef", "scaler")):
        v = _g[k]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape), str(v.dtype))
        elif isinstance(v, dict):
            print(k, "keys:", list(v)[:18])
        elif isinstance(v, (int, float, str, bool)):
            print(k, repr(v)[:160])
        elif isinstance(v, (tuple, list)):
            print(k, type(v).__name__, "length", len(v), repr(v)[:160])
        else:
            print(k, type(v).__name__)

print("\n=== IDENTIFICATION TABLES ===")
for filename in ("dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv"):
    p = Path("/work") / filename
    tab = pd.read_csv(p)
    print(filename, tab.shape)
    print(tab.head(10).to_string(index=False))

print("\n=== NOTEBOOK SECTION TITLES ===")
print("\n".join(line for line in Path("/work/notes.md").read_text().splitlines()
                if line.startswith("##")))

print("\nACTIVE EDITS:", edited())
```