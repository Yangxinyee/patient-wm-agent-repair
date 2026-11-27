```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
notes_recover136 = Path("/work/notes.md").read_text()
print("=== EARLIER NOTEBOOK ===")
print(notes_recover136[:16000])
print("=== CURRENT EDITS ===")
print(edited())
print("=== RELEVANT KERNEL VARIABLES ===")
for name136 in sorted(list(globals())):
    if name136.startswith("_"):
        continue
    if any(s136 in name136.lower() for s136 in
           ["ids", "stays", "target", "repair", "128", "134", "train", "tuning"]):
        obj136 = globals()[name136]
        if isinstance(obj136, pd.DataFrame):
            print(name136, "DataFrame", obj136.shape, list(obj136.columns))
        elif isinstance(obj136, np.ndarray):
            print(name136, "array", obj136.shape)
        elif isinstance(obj136, dict):
            print(name136, "dict keys", list(obj136.keys())[:35])
        elif isinstance(obj136, (list, tuple)):
            print(name136, type(obj136).__name__, len(obj136), repr(obj136[:110])[:1500])
        elif callable(obj136):
            print(name136, "callable")
        else:
            print(name136, type(obj136).__name__, repr(obj136)[:500])
with open("/work/notes.md", "a") as f136:
    f136.write("\n## Restart after forward 13 / run 136\n"
               "Current corrected core.b_in edit raises held-out tuning factual NLL "
               "from 1.57474 to 1.57640 (+0.00166), within +0.005. "
               "Causal rollout improvement is not yet tested. Recovering architecture, "
               "record-supported estimates, and evaluation IDs before paired rollout.\n")
```
