The architecture confirms that the installed edits change only reading predictions; their rollout effects propagate through generated readings. The notebook also shows why raw dose–outcome associations are misleading. I’ll recover the adjusted effect estimates and gate helpers next, so any stronger edit is tied to record-based evidence rather than a probe response.

```python
import inspect, re
from pathlib import Path

_n = Path("/work/notes.md").read_text()
_start = _n.find("## Run 4")
print("=== EARLY CAUSAL CHECKS ===")
print(_n[_start:_start + 4200])

print("=== TARGET / ACCEPTED-EDIT REFERENCES ===")
_hits = [line for line in _n.splitlines()
         if any(term in line.lower() for term in
                ["target", "long-run", "long run", "strength-2", "strength 2",
                 "3.290", "run 115", "run 151", "run 152"])]
for _line in _hits[:18]:
    print(_line[:450])

print("=== HELPERS ===")
for _name, _obj in list(globals().items()):
    if inspect.isfunction(_obj) and getattr(_obj, "__module__", None) == "__main__":
        try:
            print(_name, inspect.signature(_obj))
        except (TypeError, ValueError):
            print(_name)

print("=== RECENT VARIABLES ===")
for _name in sorted(list(globals())):
    if not _name.startswith(("r115", "r149", "r150", "r151", "r152")):
        continue
    _obj = globals()[_name]
    if hasattr(_obj, "shape"):
        print(_name, type(_obj).__name__, _obj.shape)
    elif isinstance(_obj, dict):
        print(_name, "dict", list(_obj)[:18])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, "len", len(_obj))
    elif isinstance(_obj, (int, float, str)):
        print(_name, repr(_obj)[:160])
    else:
        print(_name, type(_obj).__name__)
```
