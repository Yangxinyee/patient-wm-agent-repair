The stronger norepinephrine edit is installed, but installation alone proves neither factual safety nor improved dose responses. I’ll recover the model interfaces and record-derived targets before testing it. I’ll retain the trial only if it meets the precommitted factual bound and improves the rollout readouts; otherwise I’ll restore the validated checkpoint.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_r276_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD-BASED FINDINGS ===")
print(_r276_notes[:22000])

print("\n=== AVAILABLE KERNEL INTERFACES ===")
_r276_items = list(globals().items())
for _r276_name, _r276_value in _r276_items:
    if _r276_name.startswith("_"):
        continue
    if callable(_r276_value) and not isinstance(_r276_value, type):
        print("function", _r276_name)
    elif any(s in _r276_name.lower() for s in
             ("stay", "cohort", "target", "causal", "candidate", "cancel", "cache", "safe")):
        if isinstance(_r276_value, (pd.DataFrame, np.ndarray)):
            print(_r276_name, type(_r276_value).__name__, _r276_value.shape)
        elif isinstance(_r276_value, dict):
            print(_r276_name, "dict keys", list(_r276_value)[:25])
        elif isinstance(_r276_value, (list, tuple)):
            print(_r276_name, type(_r276_value).__name__, "length", len(_r276_value),
                  "first", _r276_value[:12])
        else:
            print(_r276_name, type(_r276_value).__name__,
                  str(_r276_value)[:350])

print("\n=== SAVED SAFE CHECKPOINT ===")
with np.load("/work/safe_before_ne40.npz") as _r276_safe:
    print({k: _r276_safe[k].shape for k in _r276_safe.files})
print("\n=== CURRENT EDITS ===")
print(edited())
```