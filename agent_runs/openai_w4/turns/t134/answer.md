The architecture confirms where the new channel can enter the recurrent update. The diagnostic output was truncated before the connection parameters appeared, so I’ll inspect only those objects next. A usable decoder must transfer dose changes selectively, not merely correlate with factual glucose.

```python
import inspect
import re
from pathlib import Path

_ns120 = dict(globals())
print("=== RECENT CONNECTION OBJECTS ===")
for _n120, _v120 in sorted(_ns120.items()):
    if _n120.startswith("_"):
        continue
    if re.search(r"116|117|118|causal_beta80|channel_direction112|kernel", _n120):
        if isinstance(_v120, (np.ndarray, torch.Tensor)):
            print(_n120, tuple(_v120.shape),
                  "norm", round(float(np.linalg.norm(np.asarray(_v120))), 5))
        elif isinstance(_v120, dict):
            print(_n120, "keys", list(_v120.keys())[:20])
        elif callable(_v120):
            print(_n120, inspect.signature(_v120))
        else:
            print(_n120, repr(_v120)[:300])

print("\n=== TRANSITION / READING FUNCTIONS ===")
for _n120, _v120 in sorted(_ns120.items()):
    if callable(_v120) and getattr(_v120, "__code__", None) is not None:
        print(_n120, inspect.signature(_v120))

print("\n=== SELECTED CHANNEL ===")
for _n120 in ["chosen_channel116", "causal_beta80"]:
    _v120 = _ns120.get(_n120)
    if isinstance(_v120, dict):
        for _k120, _x120 in _v120.items():
            if isinstance(_x120, (np.ndarray, torch.Tensor)):
                print(_n120, _k120, tuple(_x120.shape),
                      repr(np.asarray(_x120).reshape(-1)[:8]))
            elif isinstance(_x120, dict):
                print(_n120, _k120, "dict keys", list(_x120.keys())[:14])
            else:
                print(_n120, _k120, repr(_x120)[:450])

print("\n=== EARLY RECORD FINDINGS ===")
print(Path("/work/notes.md").read_text()[:3600])
```
