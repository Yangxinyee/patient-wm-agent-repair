The sensor gates separate the four readings well, but the arterial correction appears costly relative to our remaining factual-fit margin. I’ll recover the architecture, causal evidence, and live candidate definitions before changing the working insulin repair. A useful extension must improve norepinephrine’s response while keeping total factual degradation below 0.005 nats.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK HEAD ===")
print(notes_text[:22000])
print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes_text.splitlines() if line.startswith("#")))
print("\n=== LIVE KERNEL INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"obj", "notes_text"}:
        continue
    if isinstance(obj, np.ndarray):
        print(name, "array", obj.shape, obj.dtype)
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), "keys:", str(keys[:18])[:450])
    elif callable(obj):
        print(name, "callable")
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), str(obj[:8])[:250])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape)
print("\n=== ACTIVE EDITS ===", edited())
print("\n=== SAVED FALLBACK ===")
with np.load("/work/best_insulin_v1.npz") as saved:
    print([(k, saved[k].shape) for k in saved.files])
```
