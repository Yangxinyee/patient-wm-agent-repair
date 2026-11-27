The new channel responds to controlled inputs, but it still leaks into factual and no-drug cases. No weights are edited yet. I’ll recover the candidate’s exact interfaces and earlier causal estimates before testing full-record predictions. A useful repair must improve dose responses without exceeding the factual NLL budget.

```python
import os, json, inspect, numpy as np

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD FORMAT ===")
print(open("/data/README.md").read())
notes_text = open("/work/notes.md").read()
print("=== EARLY NOTEBOOK ===")
print(notes_text[:16000])
print("=== MOST RECENT NOTEBOOK ===")
print(notes_text[-6500:])

print("=== RETAINED FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)

print("=== RETAINED DATA / CANDIDATE OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"notes_text", "NAMES", "SHAPES"}:
        continue
    if isinstance(obj, np.ndarray):
        print(name, "array", obj.shape, obj.dtype)
    elif hasattr(obj, "shape") and not inspect.ismodule(obj):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        ks = list(obj.keys())
        print(name, "dict", len(ks), repr(ks[:12]))
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        print(name, type(obj).__name__, len(obj), type(obj[0]).__name__)
print("=== CURRENT EDITS ===", edited())
```
