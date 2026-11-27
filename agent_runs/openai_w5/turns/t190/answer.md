The notebook supports a negative insulin effect and a positive norepinephrine effect, with uncertainty about their persistence. The architecture also confirms that generated readings feed back into later predictions. The output was truncated before the diagnostic helpers appeared, so I’ll retrieve those separately to trace where dose proportionality is lost.

```python
import inspect, types, re

print("=== PERSISTENT FUNCTION SIGNATURES ===")
_helpers = []
for _n, _v in sorted(list(globals().items())):
    if inspect.isfunction(_v) and getattr(_v, "__module__", "") == "__main__":
        try:
            _s = str(inspect.signature(_v))
        except Exception:
            _s = "?"
        _helpers.append((_n, _s))
for _n, _s in _helpers:
    print(_n, _s[:180])

print("=== RELEVANT OBJECTS ===")
_pat = re.compile(r"pair|dose|rate|sensor|tilt|head|feature|state|tune|hold|layout|cand|base|gain|clean|effect|target|transport|calib", re.I)
_objlines = []
for _n, _v in sorted(list(globals().items())):
    if _n.startswith("_") or not _pat.search(_n) or callable(_v) or isinstance(_v, types.ModuleType):
        continue
    if hasattr(_v, "shape"):
        _s = f"{type(_v).__name__} {tuple(_v.shape)}"
    elif isinstance(_v, dict):
        _s = "keys=" + str(list(_v)[:18])
    elif isinstance(_v, (list, tuple)):
        _s = f"{type(_v).__name__} len={len(_v)} " + str(_v[:4])[:100]
    elif isinstance(_v, (str, int, float, bool)):
        _s = repr(_v)[:120]
    else:
        _s = type(_v).__name__
    _objlines.append(f"{_n}: {_s}")
print("\n".join(_objlines)[:6000])
print("helper_count", len(_helpers), "object_count", len(_objlines))
```
