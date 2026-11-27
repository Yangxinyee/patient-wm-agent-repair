The architecture confirms that dose effects must pass through the recurrent state, while reading predictions also use a last-value register. The early estimates support a negative insulin effect, but the norepinephrine estimates vary substantially with adjustment. I’ll recover the stronger identification tests and reconstruction helpers; stable estimates across independent checks would support a causal target.

```python
import inspect

print("=== CAUSAL DIAGNOSTICS, NOTEBOOK EXCERPT ===")
print(notes_text[4500:10500])

print("\n=== FUNCTION NAMES ===")
fn_names = [
    name for name, obj in list(globals().items())
    if not name.startswith("_") and inspect.isfunction(obj)
]
print(", ".join(fn_names))

print("\n=== SELECTED LIVE OBJECTS ===")
terms = ("state", "trans", "bundle", "memory", "graft", "checkpoint", "rd_", "panel")
for name, obj in list(globals().items()):
    if name.startswith("_") or not any(t in name.lower() for t in terms):
        continue
    if inspect.isfunction(obj):
        try:
            print(name, str(inspect.signature(obj))[:180])
        except Exception:
            pass
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:8])
    elif hasattr(obj, "shape") and not callable(getattr(obj, "shape", None)):
        print(name, "shape", obj.shape)
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, "length", len(obj))
```
