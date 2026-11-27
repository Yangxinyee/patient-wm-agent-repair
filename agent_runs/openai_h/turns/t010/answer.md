The stronger adjustments still put the one-hour effects close to zero, and the delayed estimates show no consistent response in the main sensors. This supports, but does not prove, negligible drug effects. I’ll reread the architecture to identify a narrow dose-sensitivity edit. A reproducible adjusted dose response would challenge the hypothesis; a factual NLL increase above 0.005 would disqualify an edit.

```python
from pathlib import Path
import numpy as np
import time

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DEFINITIONS ===")
print(Path("/data/README.md").read_text())
print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])
print("\n=== EXISTING WORKING OBJECTS ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(value, "shape"):
        print(key, type(value).__name__, value.shape)
    elif isinstance(value, dict):
        print(key, "dict keys:", list(value.keys())[:30])
    elif isinstance(value, (list, tuple)) and len(value) < 30:
        print(key, type(value).__name__, repr(value)[:600])
print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 7: prepare a targeted intervention\n"
            "- Stronger one-hour adjustment remains near zero in the main sensors; delayed effects are not consistently dose-responsive.\n"
            "- Rereading architecture and existing working objects before choosing a dose-invariance edit.\n"
            "- No edits yet. Python runs: 7. Forward passes: 2.\n")
```