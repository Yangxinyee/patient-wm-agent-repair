```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== RETAINED OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_"):
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns))
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), repr(keys[:14]))
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(name, "function")
print("=== CURRENT EDITS ===", edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 53\nRun 52 failed by requesting a value embedding for hospital_admission. Recovering architecture and retained objects before correcting event-type handling. No causal conclusion follows from that exception; no weights edited.\n")
```