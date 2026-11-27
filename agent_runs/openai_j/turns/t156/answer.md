The architecture confirms that the prior is evaluated before adding the code embedding and measurement register. The output was truncated before the relevant reconstruction details, so I’ll inspect those compactly. A misplaced register or incorrect prior scale would explain the implausible Monte Carlo predictions.

```python
import inspect, dis, io

print("=== FUNCTION NAMES AND PARAMETERS (NO DEFAULT VALUES) ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            params = ", ".join(inspect.signature(v).parameters)
        except Exception:
            params = "?"
        print(k + "(" + params + ")")

print("\n=== DICTIONARIES ===")
for k, v in list(globals().items()):
    if not k.startswith("_") and isinstance(v, dict):
        print(k, len(v), list(v.keys())[:5])

print("\n=== RECENT ARRAYS ===")
recent_arrays = [(k, type(v).__name__, tuple(v.shape))
                 for k, v in list(globals().items())
                 if not k.startswith("_") and hasattr(v, "shape")]
for entry in recent_arrays[-65:]:
    print(entry)

print("\n=== RECONSTRUCTION FUNCTION DETAILS ===")
for name in ("corrected_register_head", "audit_head_logits", "fn0"):
    obj = globals().get(name)
    if not inspect.isfunction(obj):
        continue
    print("\nFUNCTION", name)
    try:
        text = inspect.getsource(obj)
        print(text[:4200])
    except Exception:
        print("globals:", obj.__code__.co_names)
        print("locals:", obj.__code__.co_varnames)
        print("constants:", [x for x in obj.__code__.co_consts
                             if isinstance(x, (str, int, float, tuple))][:35])

print("\n=== RECENT SOURCE / QUERY VARIABLES ===")
for k, v in list(globals().items()):
    if isinstance(v, str) and any(s in k.lower() for s in ("src", "source", "code")):
        print(k, len(v), repr(v[:180]))
```