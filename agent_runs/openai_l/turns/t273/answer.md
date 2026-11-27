The new audit’s factual NLL increase is 0.00393 nats: within the limit, but with little room left. I’ll inspect the current candidate and fitting helpers without changing weights. Further changes must improve dose responses while staying below +0.005 on both audit sets.

```python
from pathlib import Path
import re, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECENT NOTES ===")
print(Path("/work/notes.md").read_text()[-11000:])

print("=== RECENT LIVE OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    nums = re.findall(r"\d+", name)
    if nums and max(map(int, nums)) >= 215 and not name.startswith("_"):
        shape = getattr(obj, "shape", None)
        if isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:30])
        elif shape is not None:
            print(name, type(obj).__name__, tuple(shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        elif isinstance(obj, (str, int, float, bool)):
            print(name, repr(obj)[:250])

print("=== USER FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("=== CURRENT EDITS ===")
for name in edited():
    a, b = load(name), load_original(name)
    d = a - b
    print(name, tuple(a.shape), "delta_norm", float(d.norm()), "max", float(d.abs().max()))
```
