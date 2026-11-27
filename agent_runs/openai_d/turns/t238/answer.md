```python
import os, json, numpy as np, pandas as pd, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())
print("=== EARLIER NOTEBOOK EVIDENCE ===")
_notes = open("/work/notes.md").read()
print(_notes[:18000])
print("=== RECOVERED VARIABLE INVENTORY ===")
_keys = list(globals())
for _key in _keys:
    if _key.startswith(("r199", "r200", "r201", "r202", "r203",
                        "c183", "c185", "c196", "r170", "r176", "r165",
                        "r115", "r124")):
        _v = globals()[_key]
        if hasattr(_v, "shape"):
            print(_key, type(_v).__name__, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_key, "dict", list(_v)[:35])
        elif callable(_v):
            print(_key, "callable")
        else:
            print(_key, type(_v).__name__, str(_v)[:220])
print("=== FUNCTIONS ===")
print([k for k in _keys if callable(globals()[k])
       and getattr(globals()[k], "__module__", "") == "__main__"])
print("=== CURRENT EDITS ===", edited())
```
