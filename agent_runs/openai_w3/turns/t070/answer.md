```python
from pathlib import Path
import re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK: RECORD-BASED FINDINGS ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:6500])

print("\n=== RETAINED VARIABLE INDEX ===")
items = []
for key, obj in list(globals().items()):
    if key.startswith("_") or key in {"items", "obj", "key", "notes_text"}:
        continue
    if hasattr(obj, "shape") and not callable(obj):
        items.append(f"{key}: {type(obj).__name__}{tuple(obj.shape)}")
    elif isinstance(obj, dict) and re.search(r"fit|caus|effect|dose|repair|weight|meta|state|vocab", key, re.I):
        items.append(f"{key}: dict keys={list(obj)[:12]}")
    elif callable(obj) and re.search(r"repair|edit|flow|atom|bundle|context|post|write|gate|latent|fit", key, re.I):
        items.append(f"{key}: callable")
print("\n".join(items))
print("\nCURRENT EDITS:", edited())
```
