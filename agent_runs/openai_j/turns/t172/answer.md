The record-based kernel estimates suggest stronger effects than the original rollouts: roughly −11 for doubled insulin and +4 for doubled norepinephrine on tuning stays. These estimates still depend on identification assumptions. Earlier input and posterior edits gave little improvement, so I’ll inspect the stochastic head-fitting setup before testing a more selective correction.

```python
import inspect, types

print("=== RECENT FIT OBJECTS ===")
_r156_prefixes = ("wm", "r145", "r146", "r147", "r148", "r149",
                  "r150", "r151", "r152", "r153", "repair_", "head")
for _r156_n, _r156_v in sorted(list(globals().items())):
    if not _r156_n.startswith(_r156_prefixes):
        continue
    if hasattr(_r156_v, "shape"):
        print(_r156_n, type(_r156_v).__name__, tuple(_r156_v.shape))
    elif isinstance(_r156_v, dict):
        print(_r156_n, "dict", len(_r156_v), list(_r156_v)[:8])
    elif isinstance(_r156_v, (tuple, list)):
        print(_r156_n, type(_r156_v).__name__, len(_r156_v))
    elif callable(_r156_v):
        print(_r156_n, "function", getattr(_r156_v, "__code__", None).co_varnames
              if hasattr(_r156_v, "__code__") else "")
    else:
        print(_r156_n, repr(_r156_v)[:100])

print("\n=== STOCHASTIC HELPER SOURCES ===")
for _r156_n in ("wm_mc145", "wm_loss151", "wm_mc", "repair_tilt"):
    _r156_v = globals().get(_r156_n)
    if not callable(_r156_v):
        continue
    try:
        _r156_src = inspect.getsource(_r156_v)
        print(_r156_n, "\n", _r156_src[:2500])
    except Exception as _r156_e:
        print(_r156_n, "source unavailable:", type(_r156_e).__name__)
        print("arguments:", inspect.signature(_r156_v))
        print("names:", _r156_v.__code__.co_names)
        print("constants:", repr(_r156_v.__code__.co_consts)[:700])
print("\n=== OTHER REUSABLE ARRAYS ===")
_r156_summary = [
    f"{n}:{tuple(v.shape)}"
    for n, v in sorted(list(globals().items()))
    if not n.startswith(("_",) + _r156_prefixes)
    and hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2
]
print("; ".join(_r156_summary)[:2500])
```
