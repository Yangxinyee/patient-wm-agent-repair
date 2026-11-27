The recovered diagnostics show why raw associations are insufficient: norepinephrine estimates change sign or magnitude with the adjustment, and its proposed sensor instrument is weak. I’ll locate the threshold-based insulin evidence and inspect the reconstruction code. A credible threshold estimate can guide the target; an exact local transition can guide the weight edit.

```python
import dis
import re

print("=== NOTEBOOK SECTION INDEX ===")
for match in re.finditer(r"^## .*$", notes_text, re.M):
    print(match.start(), match.group(0))

def show_function(fn, limit=3200):
    print("\nFUNCTION", fn.__name__, inspect.signature(fn))
    try:
        text = inspect.getsource(fn)
    except (OSError, TypeError):
        text = dis.Bytecode(fn).dis()
    print(text[:limit])

show_function(rec_step_with_z, 3400)
show_function(r131_step, 2200)
show_function(rec_memory, 1600)

print("\n=== RD DYNAMIC FIT ===")
print("parameters:", rd_dynamic_fit.get("x"))
print("cost:", rd_dynamic_fit.get("cost"))

print("\n=== RECONSTRUCTION OBJECTS ===")
for name, obj in list(globals().items()):
    if not name.startswith(("rec_", "rm_", "r131_")) or inspect.isfunction(obj):
        continue
    if isinstance(obj, dict):
        print(name, "dict keys:", list(obj.keys())[:5])
    elif hasattr(obj, "shape") and not callable(getattr(obj, "shape", None)):
        print(name, obj.shape)
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
```
