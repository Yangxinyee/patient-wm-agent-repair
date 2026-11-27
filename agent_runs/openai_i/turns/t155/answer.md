```python
from pathlib import Path
import numpy as np, torch, json, types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("=== SURVIVING KERNEL OBJECTS ===")
_inventory = []
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or _name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor)):
        _inventory.append((_name, type(_obj).__name__, tuple(_obj.shape)))
    elif isinstance(_obj, dict):
        _inventory.append((_name, "dict", len(_obj), list(_obj)[:8]))
    elif isinstance(_obj, (list, tuple)):
        _inventory.append((_name, type(_obj).__name__, len(_obj)))
    elif isinstance(_obj, types.FunctionType):
        _inventory.append((_name, "function", _obj.__code__.co_varnames[:_obj.__code__.co_argcount]))
    elif type(_obj).__module__.startswith("pandas"):
        _inventory.append((_name, type(_obj).__name__, getattr(_obj, "shape", None)))
for _row in sorted(_inventory):
    print(_row)

print("=== EXACT-RECONSTRUCTION NOTE CONTEXT ===")
_notes = Path("/work/notes.md").read_text()
for _needle in ["## Python 55", "## Python 56", "## Python 57",
                "## Python 58", "## Python 59", "## Python 60"]:
    _pos = _notes.find(_needle)
    if _pos >= 0:
        _end = _notes.find("\n## ", _pos + 4)
        print(_notes[_pos:_end if _end >= 0 else _pos+2500][:2500])
print("Active edits:", edited())
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 140 — reconstruction audit\n"
             "- Approximate time-context reconstruction is not approved for editing.\n"
             "- Recovering architecture and surviving helpers; next directions must be "
             "validated using actual recurrent/head responses.\n"
             "- No edits active.\n")
```
