The weights are still original. The posterior and prelude checks match, but the recurrent update and readout do not, so the current calibration cannot be trusted. I’ll re-read the equations and inspect the replay helpers before editing. Agreement with the authoritative states and predictions would validate the next step; continued disagreement would refute the replay assumptions.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== NOTEBOOK EARLY EVIDENCE ===")
notes = Path("/work/notes.md").read_text()
print(notes[:16000])
print("\n=== RECENT HELPER INVENTORY ===")
for name in sorted(list(globals())):
    if name.startswith(("r82", "r95", "r96", "r97", "r100", "r103", "r104",
                        "r105", "r107", "r112", "r113")):
        obj = globals()[name]
        if callable(obj):
            print(name, str(inspect.signature(obj)))
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj)[:18])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))
print("CURRENT EDITS", edited())
```
