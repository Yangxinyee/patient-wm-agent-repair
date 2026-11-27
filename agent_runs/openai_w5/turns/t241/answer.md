The sensor gates are correctly separated, and reconstructed outputs match the saved states. Exposure centering remains suspect, especially for lab glucose and arterial MAP, but it does not yet explain the rollout asymmetry. I’ll recover the implementation and record-derived estimates before testing whether dose changes also alter the centering term. A substantial change would support that hypothesis; stable centering would refute it.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

print("=== NOTEBOOK: IDENTIFICATION AND EFFECT EVIDENCE ===")
_notes214 = Path("/work/notes.md").read_text()
for _section214 in _notes214.split("\n## "):
    if any(s in _section214.lower() for s in [
        "identif", "instrument", "causal estimate", "record-derived",
        "effect estimate", "random", "confound", "response", "213"
    ]):
        print(("\n## " + _section214)[:6500])

print("=== KERNEL OBJECT INVENTORY ===")
_inventory214 = list(globals().items())
for _name214, _obj214 in _inventory214:
    if _name214.startswith("__") or _name214.endswith("214"):
        continue
    if isinstance(_obj214, (np.ndarray, torch.Tensor)):
        print(_name214, type(_obj214).__name__, tuple(_obj214.shape))
    elif isinstance(_obj214, dict):
        print(_name214, "dict", len(_obj214), "keys", list(_obj214)[:15])
    elif isinstance(_obj214, (list, tuple)):
        print(_name214, type(_obj214).__name__, len(_obj214))
    elif callable(_obj214) and getattr(_obj214, "__module__", None) == "__main__":
        print(_name214, "function")
print("=== CURRENT EDIT COUNT ===", len(edited()))
```
