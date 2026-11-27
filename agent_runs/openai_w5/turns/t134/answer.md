The rate-channel fit is accurate, but it has not yet demonstrated correct drug effects or preserved factual predictions. I’ll inspect the architecture, earlier effect estimates, and available helpers before installing it. The next candidate must pass both finite-dose response tests and the factual NLL limit.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:-8000] if len(notes_text) > 8000 else notes_text)
print("\n=== CURRENT EDITS ===")
print(edited())
print("\n=== PERSISTENT OBJECT INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"notes_text", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:16])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(name, "function")
    elif isinstance(obj, (list, tuple)) and len(obj) < 30:
        print(name, type(obj).__name__, repr(obj)[:240])
```
