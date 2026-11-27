The architecture confirms that drug effects enter a shared recurrent state, while the current edit changes only the value head. The output was truncated before the candidate inventory appeared. I’ll retrieve a compact inventory and the record-derived effect estimates so the next change is grounded in both.

```python
import inspect
from pathlib import Path

print("=== RECENT VARIABLES ===")
_inv = []
for _k in list(globals())[-100:]:
    if _k.startswith("_"):
        continue
    _v = globals()[_k]
    if hasattr(_v, "shape"):
        _inv.append(f"{_k}: {type(_v).__name__} {_v.shape}")
    elif isinstance(_v, dict):
        _inv.append(f"{_k}: dict keys={list(_v)[:14]}")
    elif isinstance(_v, (list, tuple)):
        _inv.append(f"{_k}: {repr(_v)[:150]}")
    elif isinstance(_v, (int, float, str)):
        _inv.append(f"{_k}: {repr(_v)[:100]}")
print("\n".join(_inv)[-4300:])

print("=== FUNCTIONS ===")
_func = []
for _k, _v in list(globals().items()):
    if not _k.startswith("_") and callable(_v) and getattr(_v, "__module__", "") == "__main__":
        try:
            _func.append(f"{_k}{inspect.signature(_v)}")
        except Exception:
            pass
print("\n".join(_func)[-3300:])

print("=== KERNEL TARGETS ===")
print(pd.read_csv("/work/kernel_targets.csv").head(12).to_string(index=False))
print("=== IV TABLE SCHEMAS ===")
for _fn in ["dynamic_iv.csv", "proxy_iv.csv", "sensor_iv.csv"]:
    _d = pd.read_csv("/work/" + _fn)
    print(_fn, _d.shape, list(_d.columns))
```
