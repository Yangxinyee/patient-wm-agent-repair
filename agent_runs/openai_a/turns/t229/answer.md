The stronger edit increased factual NLL by 0.00321 nats on these 100 stays, within the 0.005 limit. Its rollout effect is still untested. I’ll recover the architecture, saved checkpoints, and validation IDs before testing whether it strengthens the insulin response without losing factual accuracy.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

_snapshot = list(globals().items())
print("=== PERSISTENT HELPERS, CHECKPOINTS, AND ARRAYS ===")
for _name, _obj in _snapshot:
    if _name.startswith("_"):
        continue
    if callable(_obj) and getattr(_obj, "__module__", "") == "__main__":
        print("FUNCTION", _name)
    elif isinstance(_obj, (np.ndarray, torch.Tensor)):
        print("ARRAY", _name, tuple(_obj.shape), str(_obj.dtype))
    elif isinstance(_obj, dict):
        _keys = list(_obj.keys())
        print("DICT", _name, "length", len(_obj), "keys", _keys[:16])
    elif isinstance(_obj, (list, tuple)):
        print("LIST", _name, "length", len(_obj),
              "sample", repr(_obj[:8])[:240])
    elif isinstance(_obj, (str, int, float, bool)):
        print("SCALAR", _name, repr(_obj)[:240])

_notes = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK: EVIDENCE AND INITIAL CONSTRUCTION ===")
print(_notes[:18000])
print("=== CURRENT EDITS ===")
print(edited())
```
