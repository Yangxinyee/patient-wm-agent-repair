The audit shows that the projected features used to test the gates differ from the model’s actual head inputs. Earlier tests also omitted a numeric embedding term, so those gates are not reliable enough to install. I’ll recover the architecture and prior causal evidence before choosing another edit; any reconstructed pathway must first reproduce the original forward predictions.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK / CAUSAL EVIDENCE ===")
print(notes_text[:18000])
print("\n=== CURRENT EDITS ===")
print(edited())

print("\n=== PERSISTENT HELPERS ===")
for name, obj in sorted(list(globals().items())):
    if not name.startswith("_") and inspect.isfunction(obj):
        try:
            signature = str(inspect.signature(obj))
        except Exception:
            signature = "?"
        print(name, signature)

print("\n=== RELEVANT CACHE VARIABLES ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith(("rep_", "dml", "causal", "kernel", "gate", "head", "W", "df", "train", "tune")):
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(name, type(obj).__name__, tuple(shape))
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:18])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length:", len(obj))
        elif callable(obj):
            pass
        else:
            print(name, type(obj).__name__, str(obj)[:120])

print("\n=== PATHWAY TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("action", "dose", "numeric", "value", "read", "prior", "forecast", "query")):
        print(name, SHAPES[name])
```