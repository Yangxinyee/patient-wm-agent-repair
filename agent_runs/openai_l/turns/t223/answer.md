No new computation ran. The last audit showed reasonably consistent dose contrasts, but appreciable variation on factual records, so simply amplifying those directions risks worsening factual fit. I’ll recover the architecture and cached fits before constructing a nonlinear correction. The hypothesis is supported only if that correction strengthens dose responses while preserving factual predictions.

```python
from pathlib import Path
import numpy as np, torch, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLY NOTEBOOK: RECORD-BASED IDENTIFICATION ===")
notes201 = Path("/work/notes.md").read_text()
print(notes201[:22000])

print("\n=== LIVE FEATURE / FIT OBJECTS ===")
keys201 = [k for k in list(globals()) if
           any(s in k.lower() for s in
               ("192", "194", "195", "196", "197", "198", "199", "200",
                "179", "185", "gate", "kernel19", "target", "contrast", "audit"))]
for k in sorted(keys201):
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape), str(v.dtype))
    elif isinstance(v, dict):
        print(k, "dict", [(str(a), tuple(b.shape) if hasattr(b, "shape")
                          else type(b).__name__) for a, b in list(v.items())[:20]])
    elif callable(v):
        try:
            print(k, "callable", str(inspect.signature(v)))
        except Exception:
            print(k, "callable")
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v), repr(v[:4])[:500])
    else:
        print(k, repr(v)[:500])
print("\nCurrent edits:", edited())
```