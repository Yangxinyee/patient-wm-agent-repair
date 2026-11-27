The last executed run verified action-token routing, but not factual accuracy or correct dose transport through time. The rejected block added no evidence. I’ll now recover the architecture, earlier record-based effect estimates, and existing helpers so the next tests are properly specified. Accurate temporal transport and a factual NLL increase below 0.005 would support this approach; failure of either would refute it.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("=== EARLIER NOTEBOOK ===")
print(_notes[:16000])
print("Notebook characters:", len(_notes))

print("=== RELEVANT PERSISTENT OBJECTS ===")
_keys = ("caus", "effect", "coef", "truth", "rate", "channel", "priv",
         "transport", "routing", "encoder", "decode", "factual", "tune",
         "held", "response", "head", "candidate", "baseline")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or not any(k in _name.lower() for k in _keys):
        continue
    if callable(_obj):
        try:
            _desc = str(inspect.signature(_obj))
        except Exception:
            _desc = ""
    elif isinstance(_obj, dict):
        _desc = "keys=" + repr(list(_obj.keys())[:25])
    elif hasattr(_obj, "shape"):
        _desc = "shape=" + repr(tuple(_obj.shape))
    elif isinstance(_obj, (list, tuple)):
        _desc = f"len={len(_obj)} " + (repr(_obj[:12]) if len(_obj) <= 100 else "")
    else:
        _desc = repr(_obj)[:180]
    print(_name, type(_obj).__name__, _desc)

print("=== TUNING RECORD SAMPLE ===")
print(pd.read_csv("/data/tuning.csv", nrows=8).to_string(index=False))
print("=== CURRENT EDIT SHAPES ===")
for _name in edited():
    print(_name, SHAPES[_name])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart after run 125\n"
             "- Transport routing passed the local action-token selection check only.\n"
             "- No causal readout coupling is installed; factual and temporal transport "
             "validation remain required before retaining this candidate.\n")
```