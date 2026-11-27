The recalibrated directions are stable, and the weight update factorizes accurately. That supports testing the candidate, but does not establish factual safety or useful rollout effects. I’ll recover the architecture and test-stay IDs before validation; the candidate must preserve NLL and move the dose responses toward the record-based estimates.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== INSTALLED OPERATIONS ===")
print(ops())

print("\n=== TUNING IDS ===")
_tune74 = pd.read_csv("/data/tuning.csv")
print("Columns:", _tune74.columns.tolist())
_idcol74 = next(c for c in ("stay_id", "stay", "id") if c in _tune74.columns)
_ids74 = _tune74[_idcol74].drop_duplicates().tolist()
print("First 100 IDs:", _ids74[:100])

print("\n=== RELEVANT KERNEL VARIABLES ===")
for _k74 in sorted(list(globals())):
    if _k74.startswith("_"):
        continue
    if any(s in _k74.lower() for s in
           ("shear", "sensor", "candidate", "gain", "direction", "pilot",
            "target", "causal", "head", "transform", "recal", "dose")):
        _v74 = globals()[_k74]
        if isinstance(_v74, np.ndarray):
            print(_k74, "array", _v74.shape)
        elif isinstance(_v74, dict):
            print(_k74, "dict keys:", list(_v74)[:24])
        elif callable(_v74):
            print(_k74, "callable")
        elif isinstance(_v74, (str, int, float, list, tuple)):
            print(_k74, repr(_v74)[:350])

print("\n=== EARLY NOTEBOOK EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:10000])

with Path("/work/notes.md").open("a") as _f74:
    _f74.write(
        "\n## Python run 74: restart recovery\n"
        "- Re-reading architecture and recovering tuning IDs and pilot variables.\n"
        "- Current recalibrated head pilot remains installed; no edits changed.\n"
        "- Factual safety and rollout usefulness are still untested. Forward passes: 27.\n"
    )
```