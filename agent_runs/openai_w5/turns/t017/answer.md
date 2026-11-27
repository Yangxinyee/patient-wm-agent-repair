The reconstructed value head still disagrees with the forward tool, so it is not yet safe to use for designing an edit. I will re-read the architecture and inspect the saved analysis objects. Agreement after correcting the reconstruction would validate it; persistent errors would suggest missing state or sampling details.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())
print("\n=== FULL NOTEBOOK ===")
print(Path("/work/notes.md").read_text())

print("\n=== SAVED ANALYSIS OBJECTS ===")
_snapshot = list(globals().items())
for _name, _obj in _snapshot:
    if _name.startswith("_"):
        continue
    if isinstance(_obj, (pd.DataFrame, pd.Series, np.ndarray)):
        print(_name, type(_obj).__name__, _obj.shape)
    elif isinstance(_obj, dict) and _name not in ("__builtins__",):
        print(_name, "dict", "keys:", list(_obj.keys())[:25])
    elif callable(_obj) and getattr(_obj, "__module__", "") == "__main__":
        print(_name, "function")

print("\n=== TENSOR NAMES AND SHAPES ===")
for _name in NAMES:
    print(_name, SHAPES[_name])
print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Restart / run 13\n"
        "- Run 12's approximate value-head reconstruction retained meaningful "
        "mean and NLL errors; no edits are justified by that reconstruction yet.\n"
        "- Re-reading architecture and inspecting persistent analysis objects "
        "before correcting the reconstruction and checking causal estimates.\n"
    )
```