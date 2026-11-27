The last diagnostic still disagreed substantially on reading bundles, so its calibration is not safe to use for a weight edit. I’ll recover the exact architecture and retained fits. The next calibration must reproduce the exported states before I trust it.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== INITIAL NOTEBOOK EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:7000])
print("\n=== SELECTED RETAINED OBJECTS ===")
for _n in sorted(list(globals())):
    if _n.startswith("_"):
        continue
    _v = globals()[_n]
    _selected = _n.startswith(("rr50", "rr51", "r73", "r74", "rr73", "rr74"))
    if isinstance(_v, pd.DataFrame):
        print(_n, "DataFrame", _v.shape, list(_v.columns)[:12])
    elif _selected and isinstance(_v, (np.ndarray, torch.Tensor)):
        print(_n, type(_v).__name__, tuple(_v.shape))
    elif _selected and isinstance(_v, dict):
        print(_n, "dict", list(_v.keys())[:12])
    elif callable(_v) and getattr(_v, "__module__", "") == "__main__":
        print(_n, "function", getattr(getattr(_v, "__code__", None), "co_varnames", ())[:8])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 75\nReading-posterior reconstruction remains invalid. Recovering exact architecture and retained fits; no edits. Counts: 75 Python runs, 6 forward passes.\n")
```