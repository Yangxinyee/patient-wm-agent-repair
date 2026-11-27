```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("recur", "memory", "slot", "attention", "attn", "posterior",
            "value", "head", "readout", "transition", "age")):
        print(name, SHAPES[name])

print("\n=== AVAILABLE HELPER SIGNATURES ===")
_snapshot128 = dict(globals())
for name, obj in sorted(_snapshot128.items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(signature unavailable)"
        print(name, sig[:220])

print("\n=== RECENT RECONSTRUCTION OBJECTS (NO CONTENT DUMPS) ===")
for name, obj in sorted(_snapshot128.items()):
    if name.startswith(("rr_", "rp_", "rw_", "rc_")) and not inspect.isfunction(obj):
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(name, type(obj).__name__, tuple(shape))
        elif isinstance(obj, dict):
            print(name, "dict", len(obj), "keys:", list(obj)[:15])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))
        elif isinstance(obj, (str, int, float, bool)):
            print(name, repr(obj)[:180])

print("\n=== EARLY NOTEBOOK EVIDENCE ===")
_notes128 = Path("/work/notes.md").read_text()
print(_notes128[:8500])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n\n## Run 128\n"
        "Activation substitutions did not validate recurrent replay. Re-reading "
        "the documented architecture and inspecting compact helper signatures "
        "to find a structural mismatch rather than continuing activation searches. "
        "No weights edited. Counts: 128 Python runs, 6 forward passes.\n"
    )
```