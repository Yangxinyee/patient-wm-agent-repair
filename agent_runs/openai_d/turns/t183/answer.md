The support guard reduced cached factual loss while preserving the late insulin response. That is promising, but the previous GPU failure shows that cached performance alone is insufficient. I’ll recover the architecture and supporting evidence before choosing another trial; it must pass both factual and actual rollout checks.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== INITIAL EVIDENCE AND EDIT DESIGN ===")
print(notes_text[:12000])

print("\n=== PERSISTENT VARIABLE INVENTORY ===")
inventory = []
for key, value in list(globals().items()):
    if key.startswith("_") or key in {
        "In", "Out", "exit", "quit", "get_ipython", "notes_text", "inventory"
    }:
        continue
    shape = getattr(value, "shape", None)
    if shape is not None:
        desc = f"{type(value).__name__}, shape={tuple(shape)}"
    elif isinstance(value, dict):
        desc = f"dict[{len(value)}], keys={list(value)[:12]}"
    elif isinstance(value, (list, tuple)):
        desc = f"{type(value).__name__}[{len(value)}]"
    elif callable(value):
        desc = "callable"
    else:
        desc = type(value).__name__
    inventory.append(f"{key}: {desc}")
print("\n".join(sorted(inventory)))
print("\nCURRENT EDITS:", edited())
```
