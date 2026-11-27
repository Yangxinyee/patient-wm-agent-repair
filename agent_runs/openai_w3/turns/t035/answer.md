The metadata specifies strict “edges below the value” binning, including repeated drug-bin edges—a detail worth checking. The inspection stopped because it mistook NumPy’s `shape` function for an array shape; no weights changed. I’ll safely list the cached arrays and helper functions next, then check whether their binning matches that specification.

```python
import builtins
import types
import inspect
from pathlib import Path

print("=== LIVE ARRAYS, TABLES, DICTIONARIES ===")
for nm, obj in builtins.list(globals().items()):
    if nm.startswith("_") or isinstance(obj, types.ModuleType):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(nm, type(obj).__name__, builtins.tuple(obj.shape))
    elif type(obj).__name__ == "DataFrame":
        print(nm, "DataFrame", obj.shape, builtins.list(obj.columns)[:14])
    elif isinstance(obj, builtins.dict) and nm not in ("SHAPES", "VOCAB"):
        print(nm, "dict", len(obj), repr(builtins.list(obj.keys())[:8])[:180])
    elif isinstance(obj, (builtins.list, builtins.tuple)) and nm != "NAMES":
        print(nm, type(obj).__name__, len(obj))
print("=== LOCAL FUNCTIONS ===")
for nm, obj in builtins.list(globals().items()):
    if isinstance(obj, types.FunctionType) and obj.__module__ in ("__main__", None):
        try:
            print(nm, inspect.signature(obj), "names:", obj.__code__.co_names,
                  "constants:", repr(obj.__code__.co_consts)[:260])
        except Exception as exc:
            print(nm, type(exc).__name__)
print("=== NUMERIC DOSE ROW EXAMPLE ===")
if "df" in globals():
    print(df[df.event.eq("norepinephrine")].head(3).to_string(index=False))
with open("/work/notes.md", "a") as f:
    f.write("\nPython runs 29-30; forwards 3. Numeric bins use strict edges below x. Run 29 metadata printed; global inspection failed on a module's shape function. Safely recovering live arrays/helpers. No edits.\n")
```