The action embeddings now match the saved posterior means almost exactly, but later recurrent states still differ. I’ll inspect the recurrence and recover the earlier causal evidence before making another edit. Matching the recurrent states would validate synthetic dose tests; persistent errors would make those tests unreliable.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes_text[:18000])

print("\n=== AVAILABLE HELPERS AND IMPORTANT ARRAYS ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if inspect.isfunction(obj):
        print("FUNCTION", name, inspect.signature(obj))
    elif hasattr(obj, "shape") and not callable(getattr(obj, "shape", None)):
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print("ARRAY", name, shape)
print("\nCURRENT EDITS:", edited())
```
