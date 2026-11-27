The architecture shows that dose magnitude enters through the three edited embedding tensors. Removing magnitude need not remove drug identity or treatment timing. I’ll check the exact edit and recover the fitted controls; any surviving dose pathway or omitted pre-dose history would weaken the current approach.

```python
import inspect
import numpy as np
import pandas as pd
import torch

_notes = Path("/work/notes.md").read_text()
_start = _notes.find("## Nonlinear")
print("=== EARLIER DIAGNOSTICS AND EDIT ===")
print(_notes[_start:_start + 5700])

print("\n=== CACHED FUNCTIONS ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and inspect.isfunction(_obj):
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            print(_name)

print("\n=== DATAFRAMES ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and isinstance(_obj, pd.DataFrame):
        print(_name, _obj.shape, list(_obj.columns)[:35])

print("\n=== ARRAY CACHE ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and isinstance(_obj, np.ndarray) and _obj.size > 100:
        print(_name, _obj.shape)

print("\n=== EDIT AUDIT ===")
for _name in edited():
    _w = torch.as_tensor(load(_name))
    _o = torch.as_tensor(load_original(_name))
    print(_name, tuple(_w.shape),
          "delta_norm", float((_w - _o).norm()))
    if "bins" in _name:
        print("positive-row spread", float((_w[1:] - _w[1]).abs().max()),
              "absent-row delta", float((_w[0] - _o[0]).abs().max()))
    else:
        print("column norms", [round(float(v), 6) for v in _w.norm(dim=0)])
```