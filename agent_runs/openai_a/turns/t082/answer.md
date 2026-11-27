The CPU check suggests the provisional edit preserves factual fit, but its effect on dose rollouts remains untested. I’ll recover the setup and stay IDs before GPU validation. A useful edit must strengthen the insulin response while keeping the factual NLL increase below 0.005.

```python
from pathlib import Path
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())
print("\nCURRENT EDITS", edited())
print("\nTUNING IDS", np.asarray(tuning_ids)[:100].tolist())

print("\nRETAINED VARIABLES")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if isinstance(obj, np.ndarray):
        print(name, obj.shape, obj.dtype)
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(name, "function")
    elif isinstance(obj, dict) and any(s in name.lower() for s in
                                       ["graft", "fit", "causal", "rd", "decoder", "gate"]):
        print(name, "dict keys:", list(obj)[:40])

print("\nEARLY NOTEBOOK")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:18000])
```
