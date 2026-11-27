The last fit reproduced some dose changes only by introducing large factual errors, so it is not a safe edit. I’ll inspect the architecture and existing analysis objects for another weight pathway. A useful candidate must respond consistently to doses without materially worsening factual predictions.

```python
from pathlib import Path
import inspect, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:14000])
print("\n=== KERNEL OBJECT INVENTORY ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"In", "Out", "NAMES", "SHAPES", "VOCAB", "notes"}:
        continue
    if inspect.isfunction(v):
        try:
            print("FUNCTION", k, inspect.signature(v))
        except Exception:
            pass
    elif isinstance(v, np.ndarray):
        print("ARRAY", k, v.shape, str(v.dtype))
    elif hasattr(v, "shape") and not callable(v.shape):
        print("SHAPED", k, tuple(v.shape))
    elif isinstance(v, dict):
        print("DICT", k, len(v), list(v.keys())[:10])
print("CURRENT EDITS", edited())
```
