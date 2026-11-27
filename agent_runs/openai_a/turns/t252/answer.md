The last check exactly reproduced the protected scale-6 edit, so the scale-7 trial is a controlled change. It remains unaccepted. I’ll recover the validation IDs and prior evidence before testing it; a stronger insulin rollout effect with factual NLL degradation below 0.005 would support keeping it.

```python
from pathlib import Path
import pandas as pd
import inspect

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())

_audit_header = pd.read_csv("/data/tuning.csv", nrows=2)
_audit_idcol = next(c for c in ("stay_id", "stay", "id") if c in _audit_header.columns)
_audit_ids = pd.read_csv("/data/tuning.csv", usecols=[_audit_idcol])[_audit_idcol].drop_duplicates().tolist()
print("\nFIRST 100 TUNING IDS:", _audit_ids[:100])
print("\nSECOND 100 TUNING IDS:", _audit_ids[100:200])

print("\nKERNEL HELPERS")
for _audit_name, _audit_obj in list(globals().items()):
    if inspect.isfunction(_audit_obj) and getattr(_audit_obj, "__module__", "") == "__main__":
        try:
            print(_audit_name, inspect.signature(_audit_obj))
        except Exception:
            print(_audit_name)

print("\nCHECKPOINTS AND RELEVANT VARIABLES")
for _audit_name, _audit_obj in list(globals().items()):
    if _audit_name.startswith("_audit"):
        continue
    if any(s in _audit_name.lower() for s in ("checkpoint", "cache", "ridge", "gate", "contrast", "target", "rd_", "install", "whiten")):
        _audit_shape = getattr(_audit_obj, "shape", None)
        _audit_keys = list(_audit_obj.keys())[:15] if isinstance(_audit_obj, dict) else None
        print(_audit_name, type(_audit_obj).__name__, _audit_shape, _audit_keys)

print("\nEARLY NOTEBOOK EVIDENCE")
print(Path("/work/notes.md").read_text()[:16000])
print("\nCURRENT EDITS:", edited())
```