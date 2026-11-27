```python
from pathlib import Path
import inspect

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRECORD FORMAT")
print(Path("/data/README.md").read_text())

print("\nRELEVANT FUNCTIONS")
for r193_name, r193_obj in sorted(list(globals().items())):
    if inspect.isfunction(r193_obj) and any(
        s in r193_name.lower()
        for s in ("embed", "encod", "recon", "prelude", "bundle", "constraint", "feature")
    ):
        try:
            print(r193_name, inspect.signature(r193_obj))
        except Exception:
            print(r193_name)

print("\nRELEVANT SAVED OBJECTS")
for r193_name, r193_obj in sorted(list(globals().items())):
    if any(s in r193_name.lower() for s in (
        "efull", "fwd24", "187", "188", "189", "190", "191",
        "constraint_candidates", "validated_", "repair", "probe"
    )):
        if hasattr(r193_obj, "shape"):
            print(r193_name, type(r193_obj).__name__, tuple(r193_obj.shape))
        elif isinstance(r193_obj, dict):
            print(r193_name, "dict", list(r193_obj.keys())[:15])
        elif isinstance(r193_obj, (list, tuple)):
            print(r193_name, type(r193_obj).__name__, "length", len(r193_obj))
        elif inspect.isfunction(r193_obj):
            pass
        else:
            print(r193_name, type(r193_obj).__name__, str(r193_obj)[:180])

with open("/work/notes.md", "a") as r193_notes:
    r193_notes.write(
        "\n## Run193: reconstruction audit resumed\n"
        "Run192 unit sweeps did not validate independent-cohort reconstruction. "
        "Current validated_B176 remains unchanged. Recovering architecture and "
        "previously validated helper conventions before using new states in a fit.\n"
    )
```
