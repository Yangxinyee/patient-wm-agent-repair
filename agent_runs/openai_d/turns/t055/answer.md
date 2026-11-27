The reading-type gate is reliable, but the insulin decoder explains little dose variation when inactive hours are included. I won’t use that decoder directly. I’ll reread the architecture and recover the earlier causal estimates to identify a better edit pathway. A pathway that responds to dose changes while preserving factual predictions would support this approach; failure on either test would rule it out.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14000])

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
for key in sorted(list(globals())):
    if key.startswith("_") or not any(s in key.lower() for s in
        ("caus", "dml", "dose", "effect", "innov", "train", "panel", "action", "decoder", "gate")):
        continue
    value = globals()[key]
    if callable(value):
        print(key, "callable")
    elif hasattr(value, "shape"):
        print(key, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict):
        print(key, "dict keys:", list(value)[:35])
    elif isinstance(value, (list, tuple)):
        print(key, type(value).__name__, "length:", len(value), "preview:", str(value[:3])[:350])
    else:
        print(key, str(value)[:350])

print("\n=== ACTION / DYNAMICS / READING TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in ("action", "gru", "flow", "reading", "value", "last", "prior", "dose", "rate", "gate")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Run 48\nNo edits. Perfect reading-type gate, but insulin innovation decoder has only 0.029 held-out R2 when inactive hours are included. Recovering the full architecture and established causal evidence before selecting an edit pathway.\n")
```
