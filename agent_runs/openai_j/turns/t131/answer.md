The corrected contrasts show the model’s dose sensitivity, not the drugs’ causal effects. No edits are active. I’ll recover the record-based evidence and reconstruction details before fitting again. Agreement with saved GPU predictions would support the reconstruction; disagreement would require further correction.

```python
from pathlib import Path
import json, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD COLUMNS ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
audit_notes = Path("/work/notes.md").read_text()
print(audit_notes[:-8000][-18000:])
print("\n=== PERSISTENT VARIABLES ===")
for audit_name, audit_obj in list(globals().items()):
    if audit_name.startswith("_") or audit_name.startswith("audit_"):
        continue
    if isinstance(audit_obj, (pd.DataFrame, pd.Series, np.ndarray)):
        print(audit_name, type(audit_obj).__name__, audit_obj.shape)
    elif callable(audit_obj) and getattr(audit_obj, "__module__", "") == "__main__":
        print(audit_name, "function")
    elif isinstance(audit_obj, dict):
        print(audit_name, "dict", len(audit_obj), "keys:", list(audit_obj)[:8])
print("\nActive edits:", edited())
```
