The probe separates finger glucose from lab glucose well. That may let us strengthen the insulin correction without increasing lab-reading NLL. I’ll recover the architecture and edit machinery first. A selective edit that improves insulin rollouts while keeping factual NLL within the limit would confirm this approach; loss of factual fit would refute it.

```python
from pathlib import Path
import inspect, re, numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README")
print(Path("/data/README.md").read_text())

print("\nNOTEBOOK SECTION INDEX")
notes_text = Path("/work/notes.md").read_text()
for line in notes_text.splitlines():
    if line.startswith("#"):
        print(line)

print("\nRELEVANT KERNEL OBJECTS")
for name in sorted(list(globals())):
    if name.startswith("_") or name in {"notes_text"}:
        continue
    if re.search(r"sc_|r3|graft|contrast|separator|checkpoint|cache|protect|candidate|head|gate|rd_|causal", name, re.I):
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            desc = f"array {obj.shape} {obj.dtype}"
        elif callable(obj):
            try:
                desc = str(inspect.signature(obj))
            except Exception:
                desc = type(obj).__name__
        elif isinstance(obj, dict):
            desc = "dict keys=" + repr(list(obj.keys())[:25])
        else:
            desc = repr(obj)[:200]
        print(name, desc)

print("\nCURRENT EDITS", edited())
```
