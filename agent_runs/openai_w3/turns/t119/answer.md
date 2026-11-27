The helpers are still available, but their implementations are needed to resolve the mismatch. I’ll inspect the readout and update functions directly. An indexing or timing error would be fixable; poor agreement after correcting those would require a different calibration.

```python
import inspect, dis, re
import numpy as np
import torch

print("=== KEY IMPLEMENTATIONS ===")
for _name108 in ("means105", "next_view105", "parts105", "state_update82"):
    _fun108 = globals()[_name108]
    print("\nFUNCTION", _name108)
    print("GLOBAL REFERENCES", _fun108.__code__.co_names)
    try:
        print(inspect.getsource(_fun108))
    except Exception:
        dis.dis(_fun108)

print("\n=== RECENT ARRAYS / FITS ===")
for _name108 in sorted(list(globals())):
    if not re.search(r"(97|98|99|100|101|102|103|104|105)", _name108):
        continue
    _obj108 = globals()[_name108]
    if isinstance(_obj108, (np.ndarray, torch.Tensor)):
        print(_name108, type(_obj108).__name__, tuple(_obj108.shape))
    elif isinstance(_obj108, dict):
        print(_name108, "dict", list(_obj108)[:12])
    elif isinstance(_obj108, (list, tuple)):
        print(_name108, type(_obj108).__name__, len(_obj108),
              repr(_obj108[:3])[:200])
    elif not callable(_obj108):
        print(_name108, type(_obj108).__name__, repr(_obj108)[:180])

print("\n=== ORIGINAL FACTUAL FILES ===")
for _path108 in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    print(_path108)
    print(__import__("pandas").read_csv(_path108, nrows=2).to_string(index=False))
print("CURRENT EDITS", edited())
```
