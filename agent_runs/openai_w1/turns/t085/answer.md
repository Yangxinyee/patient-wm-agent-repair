The architecture confirms that doses can affect readings through the action encoder, posterior, and recurrent state. The earlier record-based estimates support negative insulin effects and positive norepinephrine effects, but they do not validate any proposed edit. I’ll inspect the retained candidates without printing long documents; their dimensions and alignment will determine whether they are usable.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_lines = []
for _n in sorted(list(globals())):
    if _n.startswith("_"):
        continue
    _v = globals()[_n]
    if isinstance(_v, pd.DataFrame):
        _lines.append(f"{_n}: DataFrame {_v.shape}, columns={list(_v.columns)[:9]}")
    elif _n.startswith(("rr50", "rr51", "rr58", "rr71", "rr72", "rr73", "rr74",
                        "r50", "r51", "r58", "r71", "r72", "r73", "r74")):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            _lines.append(f"{_n}: {type(_v).__name__} {tuple(_v.shape)}")
        elif isinstance(_v, dict):
            _lines.append(f"{_n}: dict keys={list(_v.keys())[:10]}")
        elif isinstance(_v, (list, tuple)):
            _lines.append(f"{_n}: {type(_v).__name__} length={len(_v)}")
        elif isinstance(_v, (int, float, str)):
            _lines.append(f"{_n}: {str(_v)[:150]}")
    elif callable(_v) and getattr(_v, "__module__", "") == "__main__":
        _code = getattr(_v, "__code__", None)
        _lines.append(f"{_n}: function args={_code.co_varnames[:_code.co_argcount] if _code else '?'}")
_text = "\n".join(_lines)
print("=== INVENTORY ===")
print(_text[:6200])
print("Inventory characters:", len(_text))
print("\n=== RELEVANT TENSORS ===")
for _n in NAMES:
    if _n.startswith(("encode_action.", "encode.", "z_post.", "z_prior.", "z_head.", "z_in.", "gap.")):
        print(_n, SHAPES[_n])
print("\n=== CONFIG ===")
print(VOCAB["config"])
print("VOCAB keys:", list(VOCAB))
print("Edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 76\nArchitecture recovered; run 75 output was truncated before inventory. Inspecting retained candidate objects and tensor dimensions compactly. No edits. Counts: 76 Python runs, 6 forward passes.\n")
```