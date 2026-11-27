```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:18000])
print("\n=== RETAINED OBJECTS ===")
inventory = []
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "notes_text", "inventory", "obj"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        desc = f"{type(obj).__name__}{tuple(obj.shape)}"
    elif isinstance(obj, dict):
        desc = f"dict[{len(obj)}] keys={list(obj)[:10]}"
    elif isinstance(obj, (list, tuple)):
        desc = f"{type(obj).__name__}[{len(obj)}]"
    elif isinstance(obj, (int, float, str, bool)):
        desc = repr(obj)[:90]
    elif callable(obj):
        desc = "callable"
    else:
        desc = type(obj).__name__
    inventory.append(f"{name}: {desc}")
print("\n".join(inventory))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 52: restart recovery\n"
            "Run 51 recovered insulin innovations but not norepinephrine innovations. "
            "No edit is validated or applied. Recovering causal evidence, architecture, and retained fits.\n"
            "Counts: 52 Python runs, 6 forward passes. No edits.\n")
print("CURRENT EDITS:", edited())
```