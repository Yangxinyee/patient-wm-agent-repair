The last check shows that the cached proxy omits the private dose channels, while the actual value-head input contains them. That makes exposure recovery more credible, but does not yet explain the rollout asymmetry. I’ll recover the architecture and calibration code before testing whether changed doses reach the head proportionally.

```python
import os, json, inspect, numpy as np, torch
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

_notes199 = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(_notes199[:22000])

print("\n=== PERSISTENT OBJECT INDEX ===")
_items199 = list(globals().items())
for _n199, _v199 in sorted(_items199):
    if _n199.startswith("__") or inspect.ismodule(_v199):
        continue
    if inspect.isfunction(_v199):
        print(_n199, "FUNCTION", str(inspect.signature(_v199)))
    elif isinstance(_v199, dict):
        print(_n199, "DICT", list(_v199.keys())[:24])
    elif isinstance(_v199, (np.ndarray, torch.Tensor)):
        print(_n199, type(_v199).__name__, tuple(_v199.shape))
    elif hasattr(_v199, "shape"):
        print(_n199, type(_v199).__name__, _v199.shape)
    elif isinstance(_v199, (list, tuple)):
        print(_n199, type(_v199).__name__, "len", len(_v199),
              repr(_v199[:5])[:180])
    elif isinstance(_v199, (str, int, float, bool, type(None))):
        print(_n199, type(_v199).__name__, repr(_v199)[:140])

with open("/work/notes.md", "a") as _f199:
    _f199.write("\n## Run 199: restart recovery\n"
                "- Actual value-head inputs carry private dose features; the ordinary-only proxy does not.\n"
                "- Recovering architecture, earlier evidence, and calibration objects before interpreting this difference.\n"
                "- No weights changed.\n")
```