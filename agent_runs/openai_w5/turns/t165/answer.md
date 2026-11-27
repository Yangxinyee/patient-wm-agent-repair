The recovered notes indicate negative insulin effects on glucose and positive norepinephrine effects on MAP, with much weaker original-model responses. Those estimates depend on identification assumptions. The output was truncated before the saved calibration objects appeared, so I’ll inspect them separately; this will show whether the proposed correction actually represents dose changes at the right times.

```python
import inspect, re, types
snapshot142 = list(globals().items())

print("=== RECENT PERSISTENT OBJECTS ===")
recent142 = [
    (n, o) for n, o in snapshot142
    if not n.startswith("_") and not isinstance(o, types.ModuleType)
]
for n, o in recent142[-85:]:
    if hasattr(o, "shape"):
        d = str(tuple(o.shape))
    elif isinstance(o, dict):
        d = "keys=" + repr(list(o)[:14])
    elif callable(o):
        try:
            d = str(inspect.signature(o))
        except Exception:
            d = "function"
    elif isinstance(o, (list, tuple)):
        d = f"len={len(o)} " + repr(o[:3])[:90]
    else:
        d = repr(o)[:90]
    print(n, type(o).__name__, d)

print("=== RELEVANT FUNCTION NAMES ===")
for n, o in snapshot142:
    if callable(o) and re.search(
        r"head|tilt|dose|panel|effect|causal|feature|read|state|transport|pred|rate|core",
        n, re.I
    ):
        try:
            print(n, inspect.signature(o))
        except Exception:
            print(n)

print("=== IDENTIFICATION NOTEBOOK CONTINUATION ===")
print(notes_text[6500:10500])
```
