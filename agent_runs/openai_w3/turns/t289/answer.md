The last result shows large, nonlinear calibration directions with unintended sensor effects, so another simple gain adjustment is unlikely to work. I’ll recover the architecture, record-derived estimates, and current implementation before changing weights. The next candidate must preserve factual predictions while retaining the intended dose contrasts.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLY NOTEBOOK: RECORD IDENTIFICATION ===")
notes256 = Path("/work/notes.md").read_text()
print(notes256[:15500])
print("\n=== RETAINED IMPLEMENTATION VARIABLES ===")
keys256 = [
    k for k in list(globals())
    if not k.startswith("_") and any(
        s in k.lower() for s in [
            "prelude", "replay", "checkpoint", "193", "220", "249",
            "254", "255", "target", "causal", "rate", "innovation"
        ]
    )
]
for k in keys256:
    v = globals()[k]
    if callable(v):
        try:
            print(k, "FUNCTION", inspect.getsource(v)[:4500])
        except (OSError, TypeError):
            print(k, "FUNCTION", str(inspect.signature(v)))
    elif isinstance(v, dict):
        print(k, "DICT", list(v)[:18])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    else:
        print(k, type(v).__name__, str(v)[:220])
print("\nEdited tensors:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 256; forwards 20. Restart recovery: local total-state calibration is nonlinear and has sensor crosstalk. Recovering architecture, record identification, and retained implementation before attempting a jointly fitted factual/dose-contrast correction. No weights changed.\n")
```