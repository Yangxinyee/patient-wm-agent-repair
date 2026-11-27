The assertion failure is not evidence against the edit; it means the local reconstruction needs checking first. I’ll inspect the validated computation and candidate diagnostics. Agreement with the original computation would let us proceed; a mismatch must be fixed before testing weights.

```python
import inspect, dis
from pathlib import Path

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nSUBMITTED EDITS:", edited())
for name in ("r59_core", "prior_features74"):
    obj = globals().get(name)
    if callable(obj):
        print("\nFUNCTION", name, inspect.signature(obj))
        print("GLOBALS USED:", obj.__code__.co_names)
        print(dis.Bytecode(obj).dis())

print("\nRELEVANT RETAINED OBJECTS")
for name in sorted(list(globals())):
    if name.endswith(("59", "65", "66", "73", "74")):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:15])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        elif callable(obj):
            print(name, "function", inspect.signature(obj))

if "diagnostics73" in globals():
    print("\nCANDIDATE DIAGNOSTICS73")
    print(repr(diagnostics73)[:10000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 75: reconstruction failure audit\n"
            "- Run 74 stopped at an internal assertion; it did not validate or reject an edit.\n"
            "- Recovering the exact core contract and retained candidate diagnostics before testing weights.\n"
            "- Forward passes: 4; Python runs: 75.\n")
```