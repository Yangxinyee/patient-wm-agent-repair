The last result verified the low-rank transform algebra, but not whether it leaves the model’s other pathways unchanged. Numeric gain 4 remains the only edit. I’ll inspect the architecture and earlier evidence before proceeding. Exact compensation of every non-query consumer would support a query-only edit; an uncompensated pathway would refute that assumption.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== CURRENT OPERATIONS ===")
print(ops())

notes_recover = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD-BASED EVIDENCE ===")
print(notes_recover[:11000])

print("\n=== EARLIER HEAD-TRANSFORM NOTES ===")
for anchor in ("Python run 70", "Python run 71", "Python run 72",
               "Python run 73", "Python run 74", "Python run 75"):
    pos = notes_recover.find(anchor)
    if pos >= 0:
        print(notes_recover[max(0, pos - 4):pos + 1700])

print("\n=== RELEVANT RETAINED VARIABLES ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in
           ("72", "73", "74", "probe", "causal", "target", "head", "shear",
            "direction", "transform", "effect", "stay_ids")):
        obj = globals()[name]
        if callable(obj):
            try:
                print(name, "function", inspect.signature(obj))
            except Exception:
                print(name, "callable")
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:30])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj), "sample", repr(obj[:8])[:500])
        else:
            print(name, type(obj).__name__, repr(obj)[:300])

print("\n=== HEAD / REGISTER / ENCODER SHAPES ===")
for name in NAMES:
    if any(s in name for s in ("head", "last_", "encode.mlp1", "encode_action.mlp1")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 106: query-only transform recovery\n"
            "- Numeric gain 4 remains installed; no operations changed.\n"
            "- Recovering architecture and earlier head-transform evidence before installation.\n"
            "- Need exact compensation of every non-query embedding/register consumer.\n"
            "- Forward passes used: 40; 10 remain.\n")
```