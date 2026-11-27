The retained objects contain paired original and dose-changed features, so we can fit a correction without backpropagation. Before choosing its size, I’ll recover the longitudinal estimates: matching only the next reading could still give the wrong 12-hour effect. Agreement across causal diagnostics would support a target; disagreement would require a wider uncertainty range.

```python
from pathlib import Path
import numpy as np
import torch

print("=== EARLY NOTEBOOK ===")
_notes78 = Path("/work/notes.md").read_text()
print(_notes78[:5200])

print("\n=== CAUSAL ESTIMATES ===")
for _name in ["provisional_kernels", "_moment_inputs"]:
    _obj = globals().get(_name)
    if isinstance(_obj, dict):
        print(_name)
        for _key, _val in _obj.items():
            print(" ", _key)
            if isinstance(_val, dict):
                for _k2, _v2 in _val.items():
                    if hasattr(_v2, "shape"):
                        print("   ", _k2, tuple(_v2.shape),
                              np.asarray(_v2).reshape(-1)[:8]
                              if np.prod(_v2.shape) <= 30 else "")
                    elif isinstance(_v2, (float, int, str, list, tuple)):
                        print("   ", _k2, str(_v2)[:240])
            else:
                print(str(_val)[:1200])
for _name in ["rd_refined", "table66", "tb"]:
    _obj = globals().get(_name)
    if hasattr(_obj, "to_string"):
        print(_name)
        print(_obj.to_string(index=False))

print("\n=== DATA OBJECTS ===")
_items78 = list(globals().items())
for _name, _obj in _items78:
    if _name.startswith("_") or callable(_obj):
        continue
    if hasattr(_obj, "columns") and (
        len(_obj) > 10000 or
        {"prev", "observed_bin"}.issubset(set(_obj.columns))
    ):
        print(_name, _obj.shape, list(_obj.columns)[:18])
    elif isinstance(_obj, dict) and "x" in _obj and "stay_id" in _obj:
        print(_name, "state bank", np.asarray(_obj["x"]).shape,
              "stays", len(np.unique(_obj["stay_id"])))
    elif isinstance(_obj, dict) and any(
        _s in _name.lower() for _s in ["head", "hist", "fit", "long", "moment"]
    ):
        print(_name, "dict keys", list(_obj)[:10])

print("\n=== EXISTING FORWARD FILES ===")
for _dir in sorted(Path("/work").glob("fwd_*")):
    print(_dir.name, [p.name for p in _dir.iterdir()])

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 78\n"
        "Recovered paired head perturbation arrays. Checking longitudinal "
        "record-derived targets before using immediate effects in a head repair. No edits.\n"
    )
```
