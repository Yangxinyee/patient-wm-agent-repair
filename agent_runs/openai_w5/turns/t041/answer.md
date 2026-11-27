The local encoders now closely match the saved states, including infusion-end events. That removes an important reconstruction uncertainty, but does not yet establish a repair. I’ll inspect the recurrence and available helpers next; matching its transitions to GPU states would support using it to design a dose-response edit.

```python
import inspect
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== PERSISTENT FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass
print("\n=== IMPORTANT OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj.keys())[:12])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
print("\n=== CURRENT EDITS ===", edited())
```
