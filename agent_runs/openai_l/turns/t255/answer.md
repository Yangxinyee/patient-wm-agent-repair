```python
from pathlib import Path
import numpy as np, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

notes229 = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD-EFFECT EVIDENCE ===")
print(notes229[:14000])

print("\n=== RELEVANT LIVE OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print("FUNCTION", k, inspect.signature(v))
        except Exception:
            print("FUNCTION", k)
    elif any(s in k.lower() for s in [
        "228", "227", "226", "225", "224", "223", "222",
        "219", "212", "211", "210", "209", "208", "207",
        "200", "179", "target", "kernel", "backup", "gate",
        "direction", "quiet", "audit", "canonical"
    ]):
        if isinstance(v, np.ndarray):
            print("ARRAY", k, v.shape, v.dtype)
        elif isinstance(v, dict):
            print("DICT", k, list(v.keys())[:25])
        else:
            print("OBJECT", k, type(v).__name__, str(v)[:180])
print("\nCURRENT EDITS", edited())
```
