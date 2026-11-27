```python
from pathlib import Path
import numpy as np, inspect

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
for fn in ["neutral209.npz", "corrected269.npz", "head275_0.npz"]:
    a = np.load("/work/" + fn)
    print("\nARCHIVE", fn, [(k, a[k].shape) for k in a.files])

a277 = np.load("/work/neutral209.npz")
valid277 = {k: a277[k] for k in a277.files
            if k in NAMES and tuple(a277[k].shape) == tuple(SHAPES[k])}
assert len(valid277) > 0, "Fallback archive has no recognised tensors."
for k in list(edited()):
    drop_edit(k)
for k, v in valid277.items():
    save_edit(k, v)
print("\nRESTORED", edited())

notes277 = Path("/work/notes.md").read_text()
print("\nEARLY EVIDENCE\n", notes277[:14000])
print("\nRECENT CONTEXT\n", notes277[-18000:])
print("\nFUNCTIONS", sorted(k for k, v in list(globals().items())
                           if callable(v) and getattr(v, "__module__", "") == "__main__"))
print("\nRELEVANT VARIABLES", sorted(k for k in list(globals())
      if any(s in k.lower() for s in ["target", "causal", "effect", "compens", "route", "direction", "275", "276"])))
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 277 — reject alternative-head edit\n"
            "- Forward 27 alternative-head factual dNLL +0.31001: rejected.\n"
            "- Restored neutral209; recovering architecture, causal targets, and candidate contracts.\n")
```
