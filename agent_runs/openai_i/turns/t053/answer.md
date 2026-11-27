The architecture confirms that dose information can affect the recurrent state through the latent posterior. The recovered estimates also show why raw dose–reading associations are misleading. I’ll retrieve the lag estimates and retained arrays next; we need both a defensible effect target and a usable state representation before editing weights.

```python
import re
import numpy as np
import pandas as pd

print("=== COMPACT KERNEL INVENTORY ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or _name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(_obj, "shape"):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", len(_obj))
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj))
    elif callable(_obj) and getattr(_obj, "__module__", "") == "__main__":
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            print(_name, "callable")

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?=^## )", _notes, flags=re.M)
print("\n=== CAUSAL AND LAG SECTIONS ===")
_budget = 4300
for _section in _sections:
    _heading = _section.splitlines()[0] if _section.splitlines() else ""
    if any(_word in _section.lower() for _word in ("discontinuity", "placebo", "impulse")):
        if "Python 3;" in _heading or "Python 3;" == _heading:
            continue
        _piece = _section[:min(1600, _budget)]
        print(_piece)
        _budget -= len(_piece)
        if _budget <= 0:
            break

print("\n=== POSTERIOR AND LATENT TENSORS ===")
for _name in NAMES:
    if any(_s in _name for _s in ("z_post", "z_prior", "z_in", "z_head")):
        print(_name, SHAPES[_name])
print("\nEdits:", edited())
```
