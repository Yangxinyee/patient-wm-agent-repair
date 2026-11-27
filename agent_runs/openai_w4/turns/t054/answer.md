The drug atoms now reconstruct accurately, but the measurement-posterior mismatch is still too large to trust a full-model repair. I’ll recover the architecture and retained diagnostics before proceeding. Exact reconstruction would support using the model’s internal pathways; persistent errors would argue against that approach.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_"):
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, "columns:", list(obj.columns)[:24])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and name not in {"SHAPES", "VOCAB", "__builtins__"}:
        print(name, "dict", len(obj), "keys:", list(obj.keys())[:12])
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(name, "function")

print("\n=== EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 46 (restart)\n"
            "Drug atoms reconstruct accurately; singleton measurement posterior errors remain "
            "0.07–0.12 RMS. Recovering architecture and retained diagnostics before selecting "
            "a repair. These component checks do not establish causal effects. No edits.\n")
```