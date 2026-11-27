The local reconstruction now matches the model’s recurrent state closely. That gives us a way to investigate where dose changes lose their influence. I’ll recover the relevant code and earlier evidence before choosing an edit; a useful pathway must affect later readings without substantially disturbing factual predictions.

```python
from pathlib import Path
import inspect, numpy as np
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n--- EARLIER NOTES ---")
print(Path("/work/notes.md").read_text()[:16000])
print("\n--- KERNEL INVENTORY ---")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if inspect.isfunction(v):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print("FUNCTION", k, sig)
    elif isinstance(v, np.ndarray) and v.size > 100:
        print("ARRAY", k, v.shape, v.dtype)
    elif isinstance(v, dict) and len(v) > 3:
        print("DICT", k, len(v), list(v)[:8])
print("\nCurrent edits:", edited())
```
