The recovered evidence supports an insulin effect near −1.93 per unit, with persistence around 0.83; norepinephrine’s estimate remains less securely identified. The protected edit still falls short of the insulin target. I’ll inspect the head-edit and factual caches to see whether empirical calibration can recover NLL margin without sacrificing dose sensitivity.

```python
import inspect, dis
from pathlib import Path

def _show_recovery_source(_name, _limit=2400):
    _obj = globals()[_name]
    print("\nFUNCTION", _name, inspect.signature(_obj))
    try:
        print(inspect.getsource(_obj)[:_limit])
    except (OSError, TypeError):
        _found = False
        for _key, _value in list(globals().items()):
            if isinstance(_value, str) and ("def " + _name + "(") in _value:
                _start = _value.index("def " + _name + "(")
                print("SOURCE STRING", _key, "\n", _value[_start:_start + _limit])
                _found = True
                break
        if not _found:
            print("NAMES", _obj.__code__.co_names)
            print("CONSTANTS", repr(_obj.__code__.co_consts)[:500])
            for _ins in list(dis.get_instructions(_obj))[:65]:
                print(_ins.opname, _ins.argrepr)

_show_recovery_source("make_selective_contrast_graft", 2200)
_show_recovery_source("cache_losses", 1500)

print("\nCACHES AND LATE OBJECTS")
for _name in ["cache", "r3_candidates", "trial_r3", "alt_contrast_config",
              "stronger6_checkpoint", "scale7_checkpoint", "sc_checkpoint",
              "sc_train_cache", "sc_tuning_cache"]:
    if _name not in globals():
        continue
    _obj = globals()[_name]
    print(_name)
    if isinstance(_obj, dict):
        for _key, _value in list(_obj.items())[:12]:
            if hasattr(_value, "shape"):
                print(" ", _key, type(_value).__name__, tuple(_value.shape))
            elif isinstance(_value, dict):
                print(" ", _key, "dict keys", list(_value)[:8])
            else:
                print(" ", _key, repr(_value)[:130])

print("\nCACHE-LIKE NAMES:", [
    n for n, v in list(globals().items())
    if not n.startswith("_") and isinstance(v, dict)
    and any(k in v for k in ["original_nll", "logits", "bins", "x"])
])
print("\nCURRENT R3:", {
    k: (tuple(v.shape) if hasattr(v, "shape") else v)
    for k, v in trial_r3.items()
})
```