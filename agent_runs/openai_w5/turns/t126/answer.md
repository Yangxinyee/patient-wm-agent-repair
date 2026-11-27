The last check found no retained dose-repair candidate. The four current edits only clear state coordinates, so their small factual cost does not establish a successful repair. I’ll reread the architecture and recover the earlier record-based estimates before constructing a dose channel. That route is viable only if it responds accurately to changed doses without materially disturbing factual predictions.

```python
from pathlib import Path
import re
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:11000])

_snapshot112 = list(globals().items())
print("\n=== RETAINED FUNCTIONS ===")
for _n112, _v112 in _snapshot112:
    if not _n112.startswith("_") and callable(_v112) and getattr(_v112, "__module__", "") == "__main__":
        print(_n112)

print("\n=== RELEVANT RETAINED OBJECTS ===")
_pat112 = re.compile(r"panel|coef|ridge|private|rate|effect|causal|resid|innov|candidate|fit|decoder|resp|coda|correct|state|slot", re.I)
for _n112, _v112 in _snapshot112:
    if _n112.startswith("_") or callable(_v112) or not _pat112.search(_n112):
        continue
    _shape112 = getattr(_v112, "shape", None)
    if _shape112 is not None and not callable(_shape112):
        print(_n112, type(_v112).__name__, tuple(_shape112))
    elif isinstance(_v112, dict):
        print(_n112, "dict keys:", list(_v112)[:25])
    elif isinstance(_v112, (list, tuple)):
        print(_n112, type(_v112).__name__, "length", len(_v112), "preview", repr(_v112[:4])[:350])
    elif isinstance(_v112, (str, int, float, bool)):
        print(_n112, repr(_v112)[:250])
print("\n=== EDITS ===", edited())
```
