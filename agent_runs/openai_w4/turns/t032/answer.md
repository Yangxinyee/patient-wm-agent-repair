The decoder did not generalize: its held-out innovation covariance was nearly zero. Its small estimated NLL cost therefore does not justify an edit. I’ll recover the architecture and retained analyses before testing another pathway. A useful repair must respond to dose changes—not merely preserve factual fit.

```python
from pathlib import Path
import types, inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if not name.startswith("_") and inspect.isfunction(obj):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\n=== RETAINED DATA OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:14])
    elif hasattr(obj, "shape") and not isinstance(obj, types.ModuleType):
        print(name, type(obj).__name__, tuple(obj.shape))

print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 26 (restart)\n"
            "Run-25 decoder failed independent transfer (negative innovation R2, "
            "approximately zero covariance). Low approximate factual cost does not "
            "justify that repair. Recovering architecture and retained analyses; "
            "no weights edited.\n")
```