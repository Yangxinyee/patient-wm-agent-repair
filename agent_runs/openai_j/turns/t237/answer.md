The local loss evaluator now matches the cached predictions exactly. Insulin is close to the record-based target, but doubled norepinephrine still looks too weak. I’ll inspect the existing edits and MAP contrast features before changing weights. A useful correction must strengthen the doubled-dose response while preserving factual fit.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND EDIT DESIGN ===")
notes = Path("/work/notes.md").read_text()
print(notes[:14000])

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in
           ("map19", "map20", "head177", "contrast", "tilt", "wm_train",
            "wm_tun", "kernel", "dose", "probe", "calib")):
        obj = globals()[name]
        if callable(obj):
            try:
                desc = str(inspect.signature(obj))
            except Exception:
                desc = "callable"
            code = getattr(obj, "__code__", None)
            if code is not None:
                desc += " vars=" + str(code.co_varnames)
            print(name, desc[:900])
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:35])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        else:
            print(name, type(obj).__name__, repr(obj)[:250])

print("\nActive edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart inspection, run 211\n"
            "- Run 210 verified local likelihood averaging exactly against cached outputs.\n"
            "- Calibration-base local CV/tuning value dNLL: +0.001851/+0.002267.\n"
            "- No weights changed; recovering MAP contrast features and causal evidence before a selective x2 correction.\n")
```