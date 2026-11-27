The end-marker repair fixed all bundle-count mismatches, but posterior reconstruction still has substantial error. I conclude that counting is no longer the problem; some encoding convention remains wrong. I’ll inspect the architecture, retained functions, and earlier causal evidence to isolate it. Matching posterior states across bundle types would confirm the reconstruction is safe to use for an edit.

```python
from pathlib import Path
import inspect, types, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY EVIDENCE NOTES ===")
notes = Path("/work/notes.md").read_text()
print(notes[:18000])

print("\n=== RETAINED OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith("_") or name in {"notes", "name"}:
        continue
    obj = globals()[name]
    if isinstance(obj, types.FunctionType) and obj.__module__ == "__main__":
        print("FUNCTION", name, inspect.signature(obj),
              "vars=", obj.__code__.co_varnames,
              "refs=", obj.__code__.co_names)
    elif any(s in name.lower() for s in [
        "49", "50", "51", "52", "kernel", "target", "rd_", "innov",
        "state", "r32", "action", "bundle", "d22", "train_ids"
    ]):
        desc = str(getattr(obj, "shape", ""))
        if isinstance(obj, dict):
            desc += " keys=" + str(list(obj)[:30])
        elif isinstance(obj, (list, tuple)):
            desc += " len=" + str(len(obj))
        print(type(obj).__name__, name, desc)
print("\n=== CURRENT EDITS ===", edited())
```
