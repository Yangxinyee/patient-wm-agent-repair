The needed helpers are still in the kernel, but many recent calibration objects were omitted because their names begin with underscores. I’ll retrieve those objects and the functions’ weight references. This will let me compare the dose arriving at a reading with the dose the correction subtracts as its factual baseline.

```python
import inspect, re, types

print("=== RECENT CALIBRATION AND CHANNEL OBJECTS ===")
_pat2 = re.compile(r"r14[7-9]|r15[0-9]|r16[0-3]|private|pair|reserved|strength|centering|centercoef|coords|sensor", re.I)
_lines2 = []
for _n, _v in sorted(list(globals().items())):
    if not _pat2.search(_n) or inspect.isfunction(_v) or isinstance(_v, types.ModuleType):
        continue
    if hasattr(_v, "shape"):
        _s = f"{type(_v).__name__} {tuple(_v.shape)}"
    elif isinstance(_v, dict):
        _s = "keys=" + str(list(_v)[:24])
    elif isinstance(_v, (str, int, float, bool)):
        _s = repr(_v)[:160]
    elif isinstance(_v, (list, tuple)):
        _s = f"len={len(_v)} " + str(_v[:4])[:160]
    else:
        _s = type(_v).__name__
    _lines2.append(f"{_n}: {_s}")
print("\n".join(_lines2)[:6500])

print("=== HELPER GLOBAL REFERENCES ===")
for _n in ["head_features_layout", "reconstruct_head", "exact_u162",
           "wm_step_saved", "reading_cache", "mechanistic_shifts"]:
    _f = globals()[_n]
    print(_n, "globals:", _f.__code__.co_names, "args/local:", _f.__code__.co_varnames[:14])

print("=== HEAD FEATURE IMPLEMENTATION ===")
try:
    print(inspect.getsource(head_features_layout)[:2000])
except (OSError, TypeError):
    _ret = compact161(head_features_layout, maxchars=2000)
    if _ret is not None:
        print(str(_ret)[:2000])
```
