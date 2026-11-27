The architecture confirms that the edit changes reading probabilities, not the recurrent state directly; generated readings can still amplify its effects during rollout. The inspection output was truncated before the checkpoint and stay IDs appeared. I will retrieve those details so the trial can be tested and safely reverted if factual loss exceeds the limit.

```python
import inspect
import pandas as pd
from pathlib import Path

audit_tuning_ids = (
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])
    ["stay_id"].drop_duplicates().tolist()
)
print("FIRST 100 TUNING IDS:", audit_tuning_ids[:100])

print("\nCHECKPOINTS / EDIT CONSTRUCTION:")
for _name in sorted(list(globals())):
    if _name.startswith("_"):
        continue
    if not any(k in _name.lower() for k in
               ("checkpoint", "rollback", "r3", "separator", "gate")):
        continue
    _obj = globals()[_name]
    if callable(_obj):
        try:
            _desc = str(inspect.signature(_obj))
        except Exception:
            _desc = ""
    elif isinstance(_obj, dict):
        _desc = "keys=" + repr(list(_obj)[:12])
    else:
        _desc = "shape=" + str(getattr(_obj, "shape", None))
    print(_name, type(_obj).__name__, _desc)

print("\nOTHER AVAILABLE FUNCTIONS:")
print([n for n, v in list(globals().items())
       if not n.startswith("_") and inspect.isfunction(v)])

print("\nVALUE-HEAD SHAPES:")
for _name in NAMES:
    if _name.startswith("attr_heads.value_bin"):
        print(_name, SHAPES[_name])

print("\nRECENT CAUSAL-EVIDENCE SUMMARY:")
_lines = Path("/work/notes.md").read_text().splitlines()
_matches = [
    s for s in _lines
    if any(k in s.lower() for k in
           ("rd estimate", "rd evidence", "ne flexible", "ne remains",
            "norepinephrine rd", "threshold", "persistence="))
]
for _line in _matches[-14:]:
    print(_line[:500])
```