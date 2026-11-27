The architecture confirms that dose changes enter a recurrent state and then interact with generated readings. An immediate drug coefficient therefore cannot establish the 12-hour effect. I’ll recover the record-based causal diagnostics next; consistent effects across independent dose-variation checks would support a target for the repair.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:5600])

print("\n=== DATAFRAME INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if not name.startswith("_") and isinstance(obj, pd.DataFrame):
        print(name, obj.shape)

print("\n=== FUNCTIONS ===")
print(", ".join(
    name for name, obj in sorted(list(globals().items()))
    if not name.startswith("_") and callable(obj)
    and getattr(obj, "__module__", None) == "__main__"
))

print("\n=== RETAINED DICTIONARIES ===")
for name, obj in sorted(list(globals().items())):
    if (not name.startswith("_") and isinstance(obj, dict)
            and name not in {"SHAPES", "VOCAB", "W_repair"}):
        keys = list(obj)
        print(name, len(keys), str(keys[:8])[:180])

print("\n=== OTHER NONTRIVIAL OBJECT NAMES ===")
print(", ".join(
    name for name, obj in sorted(list(globals().items()))
    if not name.startswith("_") and len(name) > 3
    and isinstance(obj, (np.ndarray, torch.Tensor))
))
```
