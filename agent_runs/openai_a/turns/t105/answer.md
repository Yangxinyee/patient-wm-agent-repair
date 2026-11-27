The reconstruction remains approximate, so synthetic dose contrasts could reflect encoding errors rather than drug effects. I’ll recover the architecture and earlier evidence before proceeding. Agreement with saved states would validate this approach; persistent discrepancies would rule it out.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

_r91_notes = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK (first 18000 characters) ===")
print(_r91_notes[:18000])

print("\n=== CURRENT EDITS ===")
print(edited())

print("\n=== AVAILABLE FUNCTIONS ===")
for _r91_name, _r91_obj in list(globals().items()):
    if inspect.isfunction(_r91_obj) and not _r91_name.startswith("_r91"):
        try:
            print(_r91_name, inspect.signature(_r91_obj))
        except Exception:
            print(_r91_name)

print("\n=== CACHED DATA / FITS ===")
for _r91_name, _r91_obj in list(globals().items()):
    if _r91_name.startswith("_") or _r91_name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(_r91_obj, "shape") and not inspect.ismodule(_r91_obj):
        print(_r91_name, type(_r91_obj).__name__, tuple(_r91_obj.shape))
    elif isinstance(_r91_obj, dict):
        print(_r91_name, "dict", list(_r91_obj.keys())[:18])
```
