The last run failed on an array-to-scalar conversion, so it provided no new evidence. The strongest tested candidate still improved insulin responses while meeting the factual-fit limit on two tuning batches. I’ll recover the edit machinery and causal estimates before making further changes; stronger effects without exceeding the NLL limit would support another edit.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_notes_recovered = Path("/work/notes.md").read_text()
print("=== NOTEBOOK HEADINGS ===")
print("\n".join(x for x in _notes_recovered.splitlines() if x.startswith("#")))
print("=== EARLY CAUSAL INVESTIGATION ===")
print(_notes_recovered[:14000])

print("=== FUNCTIONS ===")
for _name, _obj in list(globals().items()):
    if inspect.isfunction(_obj) and not _name.startswith("__"):
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            print(_name, "(signature unavailable)")

print("=== RECENT VARIABLES ===")
for _name, _obj in list(globals().items())[-110:]:
    if _name.startswith("__") or inspect.isfunction(_obj):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", list(_obj.keys())[:18])
    elif isinstance(_obj, (int, float, str, list, tuple)):
        print(_name, repr(_obj)[:260])
    else:
        print(_name, type(_obj).__name__)

print("=== CURRENT EDITS ===")
print(edited())
```
