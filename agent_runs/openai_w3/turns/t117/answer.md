The tentative drug responses have the desired signs, but the readout and core replay errors make that calibration unreliable. I’ll recover the architecture and current helpers before changing weights. Agreement with the authoritative outputs would validate the calibration; continued disagreement would refute it.

```python
import inspect
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK ===")
_notes106 = Path("/work/notes.md").read_text()
print(_notes106[:14000])

print("\n=== RECENT VARIABLES ===")
for _k106 in sorted(list(globals())):
    if _k106.startswith(("r103", "r104", "r105", "fn", "core", "readout", "prelude")):
        _v106 = globals()[_k106]
        if isinstance(_v106, np.ndarray):
            print(_k106, "array", _v106.shape, _v106.dtype)
        elif hasattr(_v106, "shape"):
            print(_k106, type(_v106).__name__, tuple(_v106.shape))
        elif isinstance(_v106, str):
            print(_k106, "string", len(_v106), _v106[:120])
        elif isinstance(_v106, dict):
            print(_k106, "dict keys", list(_v106)[:30])
        elif callable(_v106):
            try:
                print(_k106, inspect.signature(_v106))
            except Exception:
                print(_k106, "callable")
        else:
            print(_k106, type(_v106).__name__, repr(_v106)[:160])

print("\n=== TENSOR NAMES AND SHAPES ===")
for _n106 in NAMES:
    print(_n106, SHAPES[_n106])
print("\nCURRENT EDITS", edited())
```
