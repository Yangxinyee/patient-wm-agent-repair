The last executed result showed that the features can track dose changes, but they also vary on factual records, particularly for norepinephrine. That could explain why stronger corrections hurt factual fit. I’ll recover the architecture and feature helpers to test a sensor-specific nonlinear readout. The hypothesis will be supported if it strengthens dose responses without materially changing factual predictions.
```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== CURRENT EDITS ===")
for n in edited():
    a = load(n)
    b = load_original(n)
    d = np.asarray(a) - np.asarray(b)
    print(n, SHAPES[n], "delta norm", float(np.linalg.norm(d)),
          "max", float(np.max(np.abs(d))))

print("\n=== RECENT LIVE VARIABLES ===")
for n in sorted(list(globals())):
    if any(n.endswith(str(k)) for k in [173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193]):
        v = globals()[n]
        if callable(v):
            c = getattr(v, "__code__", None)
            print(n, "callable",
                  None if c is None else c.co_varnames[:c.co_argcount],
                  None if c is None else c.co_names)
        elif isinstance(v, dict):
            print(n, "dict keys", list(v.keys())[:24])
        elif hasattr(v, "shape"):
            print(n, type(v).__name__, tuple(v.shape))
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, "length", len(v))
        elif isinstance(v, (str, int, float)):
            print(n, repr(v)[:160])

print("\n=== KEY EARLIER HELPERS ===")
for n in sorted(list(globals())):
    if callable(globals()[n]) and any(s in n.lower() for s in
        ["feature", "decode", "bundle", "logit", "forward", "kernel", "head"]):
        v = globals()[n]
        c = getattr(v, "__code__", None)
        print(n, None if c is None else c.co_varnames[:c.co_argcount],
              None if c is None else c.co_names[:35])

print("\n=== NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:16000])
```
