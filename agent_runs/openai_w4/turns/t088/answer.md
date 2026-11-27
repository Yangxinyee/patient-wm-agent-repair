The inspection failed because notebook functions do not retain retrievable source text; it did not test or change the model. I’ll use their available metadata and retained arrays instead. The next useful result is the exact feature and perturbation layout needed to fit and validate a correction.

```python
import numpy as np
import torch

print("=== FUNCTION METADATA ===")
for _name in ["head_features_fixed", "intervention_bank74",
              "build_head_dataset_fixed", "transition73"]:
    _fn = globals()[_name]
    print(_name)
    print("  variables:", _fn.__code__.co_varnames)
    print("  referenced names:", _fn.__code__.co_names)
    print("  constants:", [
        _x for _x in _fn.__code__.co_consts
        if isinstance(_x, (str, int, float, tuple)) and len(str(_x)) < 260
    ])

print("\n=== VALUE-HEAD TENSORS ===")
for _name in NAMES:
    if "attr_heads.value" in _name:
        print(_name, SHAPES[_name])

print("\n=== RECENT OBJECTS ===")
_items77 = list(globals().items())
for _name, _obj in _items77[-65:]:
    if _name.startswith("_") or callable(_obj):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", len(_obj), list(_obj)[:12])
        if "74" in _name:
            for _key, _val in list(_obj.items())[:8]:
                print(" ", _key, type(_val).__name__,
                      tuple(_val.shape) if hasattr(_val, "shape") else
                      (list(_val)[:8] if isinstance(_val, dict) else ""))
    elif hasattr(_obj, "columns"):
        print(_name, "DataFrame", _obj.shape, list(_obj.columns))
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj))
    elif isinstance(_obj, (str, int, float, bool)) and len(str(_obj)) < 120:
        print(_name, repr(_obj))

print("\n=== CAUSAL OBJECT NAMES ===")
for _name, _obj in _items77:
    if any(_s in _name.lower() for _s in
           ["kernel", "causal", "moment", "response", "rd_", "effect", "longitud", "innovation"]):
        if hasattr(_obj, "shape"):
            print(_name, type(_obj).__name__, tuple(_obj.shape))
        elif isinstance(_obj, dict):
            print(_name, "dict keys", list(_obj)[:10])

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Runs 76–77\n"
        "Source inspection failed because notebook functions lack retrievable "
        "source. Recovering metadata and retained arrays instead. No edits.\n"
    )
```
