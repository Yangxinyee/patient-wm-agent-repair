The helper-name search found nothing, so I assumed the wrong naming convention. The tensor list is intact and the model remains unedited. I’ll inspect the live namespace directly and recover the retained functions; their signatures and source should reveal whether the replay mismatch is an argument or equation error.

```python
import types
print("=== LIVE FUNCTIONS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if isinstance(obj, types.FunctionType):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig, "uses", obj.__code__.co_names[:18])

print("\n=== NAMESPACE NAMES ===")
names_live = sorted(n for n in globals() if not n.startswith("__"))
print("COUNT", len(names_live))
print(" ".join(names_live)[:12500])

print("\n=== NOTEBOOK: FIRST EDIT ATTEMPTS ===")
print(notes[12000:17500])
```
