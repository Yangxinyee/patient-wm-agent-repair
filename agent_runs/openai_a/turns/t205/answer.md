```python
from pathlib import Path
print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())
print("\nNOTEBOOK")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:22000])
print("\nKERNEL VARIABLES")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"notes_text"}:
        continue
    if isinstance(v, dict):
        print(k, "dict", len(v), "keys:", list(v.keys())[:18])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v), repr(v[:8])[:250])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, (int, float, str, bool)):
        print(k, repr(v)[:250])
print("\nEDITS", edited())
```
