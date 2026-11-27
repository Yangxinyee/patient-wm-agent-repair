The architecture confirms that `core.b_in` can steer the recurrent state directly. The output was truncated before the saved fits and helper names appeared, so I’ll retrieve those compactly. I need the residual definition and readout sensitivity before constructing an edit whose factual impact we can predict.

```python
import inspect

items122 = list(globals().items())
print("=== RECENT VARIABLES ===")
for k, v in items122[-125:]:
    if k.startswith("_") or callable(v):
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:14])
    elif isinstance(v, (str, int, float, bool, type(None))):
        print(k, repr(v)[:180])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "len", len(v), repr(v[:2])[:120])
    else:
        print(k, type(v).__name__)

print("\n=== RECENT FUNCTIONS ===")
funcs122 = [(k, v) for k, v in items122
            if inspect.isfunction(v) and not k.startswith("_")]
for k, v in funcs122[-28:]:
    print(k, inspect.signature(v), "names", v.__code__.co_names[:18])

print("\n=== EFFECT AND TARGET NOTES ===")
for line122 in Path("/work/notes.md").read_text().splitlines():
    if any(s in line122.lower() for s in (
        "target", "structural", "persistence", "effect estimate",
        "per unit", "per-unit", "g-computation", "gcomp", "rollout target",
        "history estimate", "glucose slope", "map slope"
    )):
        print(line122[:800])
```
