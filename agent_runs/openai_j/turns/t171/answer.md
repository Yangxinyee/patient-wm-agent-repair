The architecture confirms that a value-head edit can leave the other prediction heads untouched. The output was truncated before the causal estimates and earlier probe results, so I’ll extract those next. They should tell us whether a selective readout remains feasible or whether the drug signal needs repair upstream.

```python
import re, types
from pathlib import Path

_r155_notes = Path("/work/notes.md").read_text()
_r155_sections = re.split(r"(?m)(?=^## )", _r155_notes)
_r155_wanted = [
    "Causal identification checks", "Dose-innovation local projections",
    "Sensor-proxy dynamic check", "Fixed-dose response-kernel estimates",
    "Run 38:", "Forward 7", "Forward passes 8", "Runs 58",
    "Forward 10", "Run 78:", "Run 115:", "Run 117:"
]
for _r155_sec in _r155_sections:
    if any(_r155_key in _r155_sec.split("\n", 1)[0] for _r155_key in _r155_wanted):
        print(_r155_sec[:1100])
print("=== AVAILABLE FUNCTIONS ===")
print(", ".join(sorted(
    n for n, v in list(globals().items())
    if not n.startswith("_") and callable(v) and not isinstance(v, type)
)))
print("=== MAJOR ARRAYS AND DICTIONARIES ===")
for _r155_n, _r155_v in sorted(list(globals().items())):
    if _r155_n.startswith("_") or isinstance(_r155_v, types.ModuleType):
        continue
    if hasattr(_r155_v, "shape") and getattr(_r155_v, "ndim", 0) >= 2:
        print(_r155_n, tuple(_r155_v.shape))
    elif isinstance(_r155_v, dict):
        print(_r155_n, "dict", len(_r155_v), list(_r155_v)[:5])
```
