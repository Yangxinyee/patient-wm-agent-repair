The architecture confirms that action edits can affect every prediction head, so a selective value-head repair may be safer. The output was truncated before the helper names appeared. I’ll recover those now and check the timing and head conventions; matching the forward predictions within sampling noise would make local tests trustworthy.

```python
import inspect
import numpy as np
import torch

print("=== USER-DEFINED FUNCTIONS ===")
for key, obj in sorted(list(globals().items())):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = ""
        print(key, sig)

print("\n=== ARRAYS, DATAFRAMES, AND CONTAINERS ===")
for key, obj in sorted(list(globals().items())):
    if key.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(key, type(obj).__name__, tuple(obj.shape))
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(key, type(obj).__name__, obj.shape,
              "columns:", list(obj.columns)[:10])
    elif isinstance(obj, dict) and key not in {"SHAPES", "VOCAB", "__builtins__"}:
        print(key, "dict", len(obj), "keys:", list(obj)[:6])
    elif isinstance(obj, (list, tuple)) and key not in {"NAMES"}:
        print(key, type(obj).__name__, len(obj))

print("\n=== HEAD/FLOW TENSORS ===")
for name in NAMES:
    if any(x in name for x in (
        "attr_heads.value", "last_", "query", "type_emb",
        "z_prior", "z_head", "gap", "flow", "out_norm", "embed.numeric"
    )):
        print(name, SHAPES[name])

print("\n=== FORWARD FILES ===")
from pathlib import Path
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
```
