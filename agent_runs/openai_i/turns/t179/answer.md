The best replay is much closer to the exposed states, but it is still approximate. The new latent directions affect the intended readings on average, with substantial variation across bundles; that does not yet establish a useful rollout correction.

I’ll recover the architecture, earlier causal evidence, and retained helpers before constructing another edit. A candidate will be supported only if matched GPU rollouts improve the dose responses while factual NLL stays within the limit.

```python
from pathlib import Path
import inspect
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK EVIDENCE ===")
print(_notes[:16000])

print("\n=== RETAINED OBJECTS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or isinstance(_obj, types.ModuleType):
        continue
    if inspect.isfunction(_obj):
        try:
            _sig = str(inspect.signature(_obj))
        except Exception:
            _sig = "?"
        print(_name, "function", _sig)
    elif isinstance(_obj, dict):
        _keys = list(_obj.keys())
        print(_name, "dict", len(_keys), "keys:", _keys[:18])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, "length", len(_obj))
    elif isinstance(_obj, (str, int, float, bool)):
        print(_name, type(_obj).__name__, repr(_obj)[:160])
    elif hasattr(_obj, "shape") and not callable(getattr(_obj, "shape")):
        print(_name, type(_obj).__name__, "shape", tuple(_obj.shape))
print("\nACTIVE EDITS:", edited())
```
