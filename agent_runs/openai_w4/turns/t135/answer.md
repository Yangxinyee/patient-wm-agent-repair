The inspection found the response directions, but stopped when it tried to compute a norm for string event labels. No weights changed. I’ll fix that diagnostic and recover the decoder and transition interfaces needed to test the connection.

```python
import inspect
from pathlib import Path

_ns121 = dict(globals())
print("=== RUN-118 OBJECTS ===")
for _n121, _v121 in sorted(_ns121.items()):
    if not _n121.startswith("_") and ("118" in _n121 or _n121 in ["chosen_channel116", "kernel_models88"]):
        if isinstance(_v121, (np.ndarray, torch.Tensor)):
            _a121 = np.asarray(_v121)
            print(_n121, _a121.shape, _a121.dtype,
                  repr(_a121.reshape(-1)[:6]))
        elif isinstance(_v121, dict):
            print(_n121, "keys", list(_v121.keys())[:20])
        elif callable(_v121):
            print(_n121, inspect.signature(_v121))
        else:
            print(_n121, repr(_v121)[:200])

print("\n=== NOTEBOOK FUNCTIONS ===")
for _n121, _v121 in sorted(_ns121.items()):
    if not _n121.startswith("_") and callable(_v121) and getattr(_v121, "__code__", None):
        print(_n121, inspect.signature(_v121))

print("\n=== CHANNEL DECODER AND KERNEL FITS ===")
for _n121 in ["chosen_channel116", "causal_beta80", "kernel_models88"]:
    _d121 = _ns121.get(_n121, {})
    for _k121, _v121 in _d121.items():
        if _k121 in ["weights", "capture"]:
            print(_n121, _k121, "keys", list(_v121.keys())[:12])
        else:
            print(_n121, _k121, repr(_v121)[:1200])

print("\n=== EARLY RECORD FINDINGS ===")
print(Path("/work/notes.md").read_text()[:3200])
```
