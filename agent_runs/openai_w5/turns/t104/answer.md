```python
from pathlib import Path
import numpy as np, torch, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14500])
print("\n=== KERNEL OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if inspect.isfunction(v):
        print("FUNCTION", k, "args", inspect.signature(v), "references", v.__code__.co_names[:35])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print("ARRAY", k, tuple(v.shape), str(v.dtype))
    elif isinstance(v, (dict, list, tuple)):
        print(type(v).__name__.upper(), k, "len", len(v), "keys/sample", list(v)[:8])
print("CURRENT EDITS", edited())
```
