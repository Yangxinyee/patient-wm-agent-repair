The architecture confirms that we can change the value head without directly changing event timing or dose prediction. The earlier reconstruction checks make controlled head-feature comparisons usable, but we still need to measure their factual cost. I’ll retrieve the feature-building code and the retained causal estimates, then test whether a small head correction can satisfy both constraints.

```python
import inspect
import numpy as np
import torch

print("=== FEATURE AND INTERVENTION CODE ===")
for _name in ["head_features_fixed", "intervention_bank74"]:
    print("\n" + _name)
    print(inspect.getsource(globals()[_name]))

print("\n=== VALUE-HEAD TENSORS ===")
for _name in NAMES:
    if "attr_heads.value" in _name:
        print(_name, SHAPES[_name])

print("\n=== LATEST OBJECTS ===")
_items76 = list(globals().items())
for _name, _obj in _items76[-45:]:
    if _name.startswith("_") or inspect.isfunction(_obj):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict keys", list(_obj)[:12])
    elif hasattr(_obj, "columns"):
        print(_name, "DataFrame", _obj.shape, list(_obj.columns))
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj))
    elif isinstance(_obj, (str, int, float, bool)) and len(str(_obj)) < 120:
        print(_name, repr(_obj))

print("\n=== CAUSAL OBJECT NAMES ===")
for _name, _obj in _items76:
    if any(_s in _name.lower() for _s in
           ["kernel", "causal", "moment", "response", "rd_", "effect", "longitud", "innovation"]):
        if hasattr(_obj, "shape"):
            print(_name, type(_obj).__name__, tuple(_obj.shape))
        elif isinstance(_obj, dict):
            print(_name, "dict keys", list(_obj)[:10])

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 76\n"
        "Recovered architecture: value-head repair can leave timing and action "
        "heads unchanged directly, but recurrent sampled-value feedback still "
        "requires rollout validation. Retrieving retained feature and causal "
        "objects for an analytically fitted, factual-regularized correction. No edits.\n"
    )
```
