The inspection was truncated because a function signature included cached tensors; it revealed no new model result. I’ll limit the output to function names, graft metadata, and the original identification notes. Those are needed to scale the edit correctly and verify that its target still rests on the records.

```python
import inspect
from pathlib import Path

print("FUNCTION NAMES AND PARAMETER NAMES")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, list(inspect.signature(obj).parameters))
        except Exception:
            print(name)

print("\nGRAFT METADATA")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in
           ["graft", "robust", "beta", "pool3", "decoder3", "gate3"]):
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape)
        elif isinstance(obj, dict):
            print(name, "keys", list(obj.keys())[:12])
        elif isinstance(obj, (str, int, float, bool, list, tuple)):
            print(name, repr(obj)[:150])

print("\nEARLY IDENTIFICATION NOTES")
print(Path("/work/notes.md").read_text()[:5600])
```
