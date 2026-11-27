The prelude retains some treatment information, but the tested corrections distort factual cases too much. I will not use them. Next I’ll inspect the architecture and existing helpers for a more isolated dose pathway. Any candidate must handle finite dose changes consistently and pass the factual-fit constraint.

```python
from pathlib import Path
import inspect
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD COLUMNS ===")
print(Path("/data/README.md").read_text())

print("=== TENSOR LAYOUT ===")
for name in NAMES:
    print(name, SHAPES[name])

print("=== LOCAL FUNCTION SIGNATURES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, types.FunctionType) and obj.__module__ == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("=== PERSISTENT OBJECT INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or isinstance(obj, (types.ModuleType, types.FunctionType, type)):
        continue
    shape = getattr(obj, "shape", None)
    if shape is not None and not callable(shape):
        print(name, type(obj).__name__, tuple(shape))
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), "keys:", repr(keys[:12])[:500])
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        print(name, type(obj).__name__, len(obj), "first:", repr(obj[0])[:180])

print("=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart after run 99\n"
        "- No edits retained. Prelude innovation decoding has held-out R2 about .20, "
        "but finite-dose correction fits caused excessive factual variation.\n"
        "- Reinspect exact architecture and persistent helpers before choosing an isolated "
        "dose pathway. All candidates require finite-dose and paired GPU factual validation.\n"
    )
```