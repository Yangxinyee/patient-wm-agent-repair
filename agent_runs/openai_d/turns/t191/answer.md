The guarded insulin edit passed the latest factual check and produced shifts close to the provisional record-derived targets. The NE gate separates most treated readings, but its earlier factual loss remains unresolved. I’ll recover the architecture and fitting setup before editing further; the next NE candidate must improve dose responses without exhausting the factual-loss allowance.

```python
from pathlib import Path
import inspect, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?m)(?=^## )", _notes)
print("\n=== EARLY NOTEBOOK / IDENTIFICATION EVIDENCE ===")
print("".join(_sections[:10])[:22000])
print("\n=== RECENT METHODS ===")
for _sec in _sections:
    if re.match(r"## .*?(?:Run (?:10[0-9]|11[0-9]|13[0-9]|15[0-9]|16[0-5])\b)", _sec):
        print(_sec[:2200])

print("\n=== KERNEL FUNCTIONS ===")
for _name, _obj in sorted(list(globals().items())):
    if inspect.isfunction(_obj) and getattr(_obj, "__module__", "") == "__main__":
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            print(_name)

print("\n=== RELEVANT KERNEL VARIABLES ===")
for _name, _obj in sorted(list(globals().items())):
    if (re.search(r"ne|ridge|gate|pair|causal|target|head|warm|guard", _name, re.I)
        or re.match(r"r(?:11[0-9]|13[0-9]|15[0-9]|16[0-5])_", _name)):
        if inspect.isfunction(_obj) or inspect.ismodule(_obj):
            continue
        _desc = f"shape={_obj.shape}" if hasattr(_obj, "shape") else ""
        if isinstance(_obj, dict):
            _desc = f"keys={list(_obj)[:35]}"
        elif isinstance(_obj, (list, tuple)):
            _desc = f"len={len(_obj)}"
        elif isinstance(_obj, (str, int, float, bool, type(None))):
            _desc = repr(_obj)[:220]
        print(_name, type(_obj).__name__, _desc)
print("\nCURRENT EDITS:", edited())
```
