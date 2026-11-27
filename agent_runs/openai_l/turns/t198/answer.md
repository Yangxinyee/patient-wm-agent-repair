The quiet write has preserved factual fit, but its dose signal is small relative to ordinary variation at the value head. I’ll recover the architecture and wire details before testing a stronger route. It would be promising only if dose sensitivity improves without exceeding the factual-loss limit.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README\n", Path("/data/README.md").read_text())
print("\nCURRENT EDITS", edited())

keys181 = [
    k for k in list(globals())
    if not k.startswith("_") and (
        any(s in k.lower() for s in ["167", "173", "174", "178", "179", "180", "118", "152", "kernel", "head_input"])
        or k in ["w63_cache", "records", "train", "tuning", "df", "VOCAB", "NAMES"]
    )
]
print("\nRECOVERY VARIABLES")
for k in sorted(keys181):
    v = globals()[k]
    if callable(v):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, "FUNCTION", sig)
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "DICT", list(v)[:24])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "len", len(v), "preview", str(v[:4])[:400])
    else:
        print(k, type(v).__name__, str(v)[:500])

notes181 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-EFFECT EVIDENCE")
for s in notes181.split("\n## "):
    if any(w in s[:160].lower() for w in ["effect", "identif", "regression", "kernel", "discontinu", "data", "causal", "lag"]):
        print("\n##", s[:1800])
print("\nRECENT WIRE DETAILS\n", notes181[-12500:])
```
