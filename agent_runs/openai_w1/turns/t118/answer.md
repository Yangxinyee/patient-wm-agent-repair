The recent objects are not named with the prefixes I expected, so that inspection recovered no implementation details. I’ll locate the actual kernel variables and recover the original record-based findings. This will let me reuse validated work rather than repeat it or mix incompatible caches.

```python
import re, inspect

print("=== ORIGINAL NOTEBOOK FINDINGS ===")
print(pathlib.Path("/work/notes.md").read_text()[:4800])

print("\n=== RECENT GLOBAL OBJECTS (COMPACT) ===")
_s107_items = list(globals().items())
for name, obj in _s107_items[-155:]:
    if name.startswith("_"):
        continue
    if callable(obj) and hasattr(obj, "__code__"):
        print(name, "function", str(inspect.signature(obj)),
              "names=", obj.__code__.co_names[:18])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys=", repr(list(obj)[:5])[:140])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:90])
    else:
        print(name, type(obj).__name__)

print("\n=== USER-DEFINED FUNCTION NAMES ===")
print([n for n, o in _s107_items if callable(o) and hasattr(o, "__code__")])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 107\nRecent objects use different names than expected. Locating actual kernel objects and recovering original causal findings; no edits. Counts: 107 Python runs, 6 forward passes.\n")
```