The stronger edit fits the local dose contrasts better, but its factual fit and rollout effects are still untested. I’ll recover the exact test stays and saved candidates before testing factual fit. A value-NLL increase above 0.005 nats would reject it regardless of rollout accuracy.

```python
import json, re, inspect
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())
print("\nDATA SCHEMA")
print(open("/data/README.md").read())

_cols286 = pd.read_csv("/data/tuning.csv", nrows=2).columns.tolist()
_idcol286 = next(c for c in _cols286 if "stay" in c.lower())
_tids286 = pd.read_csv("/data/tuning.csv", usecols=[_idcol286])[_idcol286].drop_duplicates().tolist()
print("\nTUNING FIRST 100 IDS")
print(json.dumps(_tids286[:100]))
print("\nRECENT KERNEL NAMES")
print(list(globals().keys())[-110:])
print("\nSAVED CHECKPOINT INVENTORY")
for _k286, _v286 in list(globals().items()):
    if re.search(r"checkpoint(222|267|273|284|285)|cand.*267|267.*cand|target.*271|effect.*271", _k286, re.I):
        print(_k286, type(_v286).__name__,
              list(_v286.keys())[:20] if isinstance(_v286, dict) else getattr(_v286, "shape", None))
print("\nEDITED", edited())
with open("/work/notes.md", "a") as _f286:
    _f286.write("\nPython run 286; forwards 26. Restart recovery: reread architecture, recovered exact tuning IDs and recent candidate/checkpoint inventory. Gamma=10 total-response candidate remains installed; no weights changed. Factual safeguard is next.\n")
```