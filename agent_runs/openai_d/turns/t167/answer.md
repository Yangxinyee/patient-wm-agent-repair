The reconstruction validates the local loss calculation, but the edit still failed the factual-fit limit on different stays. I’ll recover the causal evidence and exact edit definitions before revising it. The next candidate must improve held-out factual fit without losing the record-supported dose response.

```python
from pathlib import Path
import re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())

_recover_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK HEADINGS ===")
for _line in _recover_notes.splitlines():
    if _line.startswith("#"):
        print(_line)
print("=== EARLY NOTEBOOK: CAUSAL EVIDENCE AND BASELINE ===")
print(_recover_notes[:18000])

print("=== KERNEL INVENTORY ===")
for _name in sorted(list(globals())):
    if _name.startswith("_") or _name in ("NAMES", "SHAPES", "VOCAB"):
        continue
    _obj = globals()[_name]
    if callable(_obj):
        print(_name, "callable")
    elif hasattr(_obj, "shape"):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", list(_obj.keys())[:18])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj))
    elif isinstance(_obj, (str, int, float, bool, Path)):
        print(_name, repr(_obj)[:160])
print("=== FORWARD FILES ===")
for _dir in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[1])):
    print(_dir.name, [p.name for p in _dir.iterdir()])
```
