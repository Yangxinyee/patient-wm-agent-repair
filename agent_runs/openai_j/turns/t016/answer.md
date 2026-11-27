Norepinephrine’s estimated response is reasonably consistent, but insulin’s long-term response remains uncertain—the bootstrap even changes sign. I’ll treat the rollout targets as provisional. Next I’ll inspect the architecture and retained objects to identify a selective, compensable weight edit. A distinct drug-response pathway would support that approach; entangled pathways would require more careful factual-fit checks.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
for key, value in VOCAB.items():
    if isinstance(value, dict):
        print(key, "dict keys:", list(value)[:30])
    elif isinstance(value, list):
        print(key, "list length:", len(value), "sample:", value[:4])
    else:
        print(key, repr(value)[:300])

print("\n=== RETAINED OBJECTS ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(value, "shape"):
        print(key, type(value).__name__, value.shape)
    elif isinstance(value, dict):
        print(key, "dict", "keys:", list(value)[:15])
    elif isinstance(value, (list, tuple)) and len(value) > 0:
        print(key, type(value).__name__, "length:", len(value))
    elif isinstance(value, (str, int, float, bool)) and len(str(value)) < 180:
        print(key, repr(value))

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
```
