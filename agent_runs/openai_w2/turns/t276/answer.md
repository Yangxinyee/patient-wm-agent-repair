The blend strengthens the half-dose response while preserving the doubled-dose probe, but it also increases factual disturbance. That is not yet evidence of an improvement. I’ll recover the exact edit construction and record-based effect estimates before testing it; only rollouts and factual NLL can confirm its value.

```python
from pathlib import Path
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes228 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-EFFECT EVIDENCE")
print(notes228[:12500])

print("\nRECENT CACHED OBJECTS")
for key228 in sorted(list(globals())):
    if any(tag228 in key228 for tag228 in
           ("227", "226", "225", "218", "212", "147", "171")):
        obj228 = globals()[key228]
        if isinstance(obj228, np.ndarray):
            print(key228, "array", obj228.shape, obj228.dtype)
        elif hasattr(obj228, "shape"):
            print(key228, type(obj228).__name__, tuple(obj228.shape))
        elif isinstance(obj228, dict):
            print(key228, "dict keys", list(obj228)[:20])
        elif isinstance(obj228, (int, float, str, tuple, list)):
            text228 = repr(obj228)
            print(key228, text228[:220])

print("\nCURRENT EDITS", edited())
```
