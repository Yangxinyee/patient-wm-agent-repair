```python
import os, inspect, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

_snapshot = list(globals().items())
print("=== RETAINED FUNCTIONS ===")
for _n, _v in _snapshot:
    if not _n.startswith("_") and inspect.isfunction(_v):
        try:
            print(_n, str(inspect.signature(_v)))
        except Exception:
            print(_n)

print("=== RETAINED DATAFRAMES ===")
for _n, _v in _snapshot:
    if not _n.startswith("_") and isinstance(_v, pd.DataFrame):
        print(_n, _v.shape, list(_v.columns)[:22])

print("=== RETAINED LARGE ARRAYS ===")
for _n, _v in _snapshot:
    if not _n.startswith("_") and isinstance(_v, (np.ndarray, torch.Tensor)) and _v.size if False else False:
        pass
for _n, _v in _snapshot:
    if not _n.startswith("_") and isinstance(_v, (np.ndarray, torch.Tensor)):
        if np.prod(_v.shape) >= 500:
            print(_n, tuple(_v.shape))

print("=== RETAINED DICTIONARIES ===")
for _n, _v in _snapshot:
    if not _n.startswith("_") and isinstance(_v, dict) and _n not in ("VOCAB", "SHAPES"):
        print(_n, "keys:", list(_v.keys())[:24])

print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 68 (restart)\nRun-67 reconstructed recurrent transition matches exported h (RMSE 1e-5) and out (RMSE 1.4e-7) with all current events, unnormalised memory, and both age updates. This validates forward reconstruction, not causal identification. Recovering retained functions/data to design and validate a targeted repair. No weights edited.\n")
```