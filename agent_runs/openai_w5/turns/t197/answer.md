The architecture confirms that doses pass through several state transformations before reaching the value head. Matching only the private coordinates is therefore insufficient. The output was truncated before the helper inventory, so I’ll retrieve that separately and locate the remaining reconstruction mismatch. Only a full-state match would justify using the CPU emulator for calibration.

```python
import inspect, types
items = sorted(list(globals().items()))
print("=== WM HELPERS AND SETTINGS ===")
for k, v in items:
    if k.startswith("wm_"):
        if inspect.isfunction(v):
            print(k, str(inspect.signature(v)))
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), list(v)[:5])
        else:
            print(k, str(v)[:180])

print("\n=== OTHER FUNCTION NAMES ===")
print(" | ".join(k + str(inspect.signature(v)) for k, v in items
                 if inspect.isfunction(v) and not k.startswith(("wm_", "_"))))

print("\n=== DICTIONARIES ===")
for k, v in items:
    if isinstance(v, dict) and not k.startswith("_"):
        print(k, len(v), str(list(v)[:5])[:150])

print("\n=== ARRAY / TABLE INVENTORY ===")
lines = [f"{k}: {type(v).__name__}{tuple(v.shape)}"
         for k, v in items if hasattr(v, "shape") and not k.startswith("_")]
print("\n".join(lines)[:4800])
```
