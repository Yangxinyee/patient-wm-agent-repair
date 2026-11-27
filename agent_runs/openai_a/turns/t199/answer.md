The edit strengthened both insulin responses to −3.61 and +1.57, but they remain well short of the record-derived estimates. Norepinephrine responses barely changed. I’ll recover the causal evidence and edit construction before choosing a stronger candidate; it must improve insulin rollouts without exhausting the factual-fit margin.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

nt = Path("/work/notes.md").read_text()
print("=== NOTEBOOK BEGINNING ===")
print(nt[:26000])
print("=== NOTEBOOK SECTION INDEX ===")
for line in nt.splitlines():
    if line.startswith("#"):
        print(line)

print("=== PERSISTENT KERNEL OBJECTS ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"nt", "value"}:
        continue
    if isinstance(value, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(key, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict):
        print(key, "dict", list(value.keys())[:18])
    elif callable(value) and getattr(value, "__module__", "") == "__main__":
        print(key, "function")

with open("/work/notes.md", "a") as f:
    f.write("\n## Protected-contrast GPU validation; Python 179, forwards 13/50\n"
            "- Current r3 lambda=0.1, scale=1 candidate: factual tuning GPU NLL "
            "1.63536 versus original 1.63461, delta +0.00075 on 4,751 readings.\n"
            "- Paired tuning rollout (100 stays): insulin x2 -3.613, half +1.566; "
            "original -1.847, +0.777. Norepinephrine x2 -0.526, half -0.608; "
            "original -0.498, -0.600.\n"
            "- Directional improvement but insulin effects still too weak. "
            "Recovering causal evidence and edit machinery before further scaling.\n")
```
