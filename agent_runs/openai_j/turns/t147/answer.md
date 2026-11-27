The candidate badly damaged factual predictions, so the rollback was necessary. No edits remain. I’ll check the reconstruction against the model specification before trying another repair; agreement on the rejected candidate’s first readings would support the reconstruction, while disagreement would expose the missing input.

```python
from pathlib import Path
import inspect

assert edited() == [], "Unexpected active edits"
print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== SAVED FORWARD FILES ===")
for p in sorted(Path("/work").glob("fwd_*"),
                key=lambda p: int(p.name.split("_")[-1])):
    print(p.name, [x.name for x in p.iterdir()])

print("\n=== RECONSTRUCTION FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and any(
        s in name.lower()
        for s in ("repair", "reconstruct", "numeric", "register", "head", "bundle")
    ):
        print(name, inspect.signature(obj))
        if name in ("repair_base_head", "repair_centres_for_meta"):
            try:
                print(inspect.getsource(obj))
            except (OSError, TypeError):
                pass

print("\n=== RELEVANT KERNEL OBJECTS ===")
for name, obj in list(globals().items()):
    if any(s in name.lower() for s in (
        "candidate", "compress", "gate", "contrast", "head_meta",
        "base_head", "state", "factual", "tuning", "rejected"
    )) and not inspect.isfunction(obj):
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(name, type(obj).__name__, tuple(shape))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj)[:25])
```
