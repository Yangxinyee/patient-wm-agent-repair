The activity score barely changes when insulin is doubled or halved in the synthetic examples, so the gate alone does not explain the weak halving response. I’ll recover the causal estimates and edit construction before changing weights. If the correction itself is asymmetric, that would support a decoder explanation; otherwise, the rollout needs closer examination.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY CAUSAL AND EDIT NOTES ===")
print(_notes[:18000])

print("\n=== RELEVANT GLOBALS ===")
_keys = [k for k in globals() if not k.startswith("__") and
         any(s in k.lower() for s in
             ["checkpoint", "gated", "activity", "install", "build", "r3",
              "causal", "rd_", "target", "insulin", "norepi", "cache"])]
for k in sorted(_keys):
    v = globals()[k]
    if callable(v):
        print(k, "function", inspect.signature(v))
    elif isinstance(v, np.ndarray):
        print(k, "array", v.shape)
    elif isinstance(v, dict):
        print(k, "dict keys", list(v)[:18])
    elif isinstance(v, (str, int, float, list, tuple)):
        print(k, repr(v)[:350])
    else:
        print(k, type(v).__name__)

print("\n=== EDIT CONSTRUCTION HELPERS ===")
for k in ["selective_cache", "install_gated_candidate", "install_candidate",
          "build_gated_candidate", "cache_losses", "sc_evaluate"]:
    if k in globals() and callable(globals()[k]):
        f = globals()[k]
        try:
            print(k, inspect.getsource(f))
        except (OSError, TypeError):
            print(k, "globals:", f.__code__.co_names,
                  "locals:", f.__code__.co_varnames,
                  "constants:", repr(f.__code__.co_consts)[:1800])
print("\nCurrent edits:", edited())
```
