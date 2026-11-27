The architecture confirms that dose effects enter through recurrent state and then the value head. The output was cut off before the live fitting helpers appeared, so I’ll inspect those compactly next. No weights will change.

```python
import re, inspect
from pathlib import Path

print("=== NOTEBOOK END ===")
print(Path("/work/notes.md").read_text()[-1800:])

recent244 = []
for name, obj in sorted(list(globals().items())):
    nums = re.findall(r"\d+", name)
    if nums and max(map(int, nums)) >= 225 and not name.startswith("_"):
        if hasattr(obj, "shape"):
            desc = str(tuple(obj.shape))
        elif isinstance(obj, dict):
            desc = "keys=" + str(list(obj.keys())[:12])
        elif isinstance(obj, (list, tuple)):
            desc = "len=" + str(len(obj))
        else:
            desc = repr(obj)[:120]
        recent244.append(name + ": " + desc)
print("=== RECENT OBJECTS ===")
print("\n".join(recent244)[:4200])

print("=== RECENT FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        if any(s in name.lower() for s in ["fit", "head", "stage", "eval", "pair", "nll", "raw", "feature"]):
            print(name, inspect.signature(obj))

print("=== CURRENT EDITS ===")
for name in edited():
    d = load(name) - load_original(name)
    print(name, "norm", float(d.norm()), "max", float(d.abs().max()))
```
