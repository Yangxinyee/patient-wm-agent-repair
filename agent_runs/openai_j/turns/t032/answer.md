Rebuilding the inputs reproduced the earlier results, so stale variables are not the cause of the mismatch. Doubling the time flow helped, but the remaining bias is too large to trust for weight edits. I’ll re-read the architecture and inspect the saved helpers; the documented computation should match the forward tool within its measured sampling noise.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLIER NOTEBOOK CONTEXT ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:14000])

print("\n=== KERNEL VARIABLES ===")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"notes_text", "key"}:
        continue
    obj = globals()[key]
    if isinstance(obj, np.ndarray):
        print(key, "ndarray", obj.shape, obj.dtype)
    elif isinstance(obj, torch.Tensor):
        print(key, "tensor", tuple(obj.shape), obj.dtype)
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function", getattr(getattr(obj, "__code__", None), "co_varnames", ())[:12])
    elif isinstance(obj, (dict, list, tuple)):
        preview = list(obj.keys())[:8] if isinstance(obj, dict) else None
        print(key, type(obj).__name__, "len", len(obj), "keys", preview)
    elif hasattr(obj, "shape"):
        print(key, type(obj).__name__, obj.shape)

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n- Run 26: fresh reconstruction exactly reproduced existing inputs. Documented flow still gives scaled mean RMSE 0.114 and bias 0.054; doubled flow improves RMSE to 0.097 but is not yet justified. No edits. Re-reading architecture and recovering kernel context before proceeding.\n")
```