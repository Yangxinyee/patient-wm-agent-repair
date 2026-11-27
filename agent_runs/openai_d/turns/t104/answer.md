The architecture confirms that the gap network is named `gap`, not `core.gap`. The output was truncated before showing the saved diagnostics. I’ll retrieve those compactly, then finish checking the controlled probe. If it cannot identify the register inputs consistently, I’ll stop relying on that reconstruction.

```python
from pathlib import Path
import numpy as np

print("=== GLOBAL NAMES ===")
gn = sorted(n for n in globals() if not n.startswith("_"))
print(" ".join(gn))

print("\n=== RECENT DIAGNOSTIC VARIABLES ===")
for n in gn:
    if n.startswith(("r87", "r88", "r89", "r90", "r91")):
        v = globals()[n]
        if hasattr(v, "shape"):
            print(n, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "dict", len(v), repr(list(v.keys())[:12]))
        elif isinstance(v, (list, tuple)):
            print(n, repr(v)[:250])
        elif callable(v):
            print(n, "function")
        else:
            print(n, repr(v)[:200])

print("\n=== REGISTER AND GAP WEIGHTS ===")
for n in NAMES:
    if n.startswith(("last_", "gap.", "type_emb", "embed.numeric")):
        print(n, SHAPES[n])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))

print("\n=== EARLY CAUSAL NOTES ===")
notes = Path("/work/notes.md").read_text()
print(notes[:5200])

print("\n=== PROBE ARTIFACT FILES ===")
print([p.name for p in Path("/work/fwd_9").iterdir()])
print("CURRENT EDITS:", edited())
```
