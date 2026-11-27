The adjusted record analyses support insulin lowering glucose and norepinephrine raising MAP. The cumulative targets remain provisional. I’ll inspect the cached construction without printing large tensor defaults, then choose a candidate that addresses factual outliers. It must reduce loss beyond the fitted states, not merely look better locally.

```python
import inspect, hashlib
import numpy as np

print("=== HELPER ARGUMENT NAMES ===")
for _name, _obj in list(globals().items()):
    if inspect.isfunction(_obj) and getattr(_obj, "__module__", None) == "__main__":
        try:
            print(_name, [p.name for p in inspect.signature(_obj).parameters.values()])
        except (TypeError, ValueError):
            print(_name)

print("=== EDIT AND GATE CACHE METADATA ===")
for _name in sorted(list(globals())):
    if not _name.startswith(("r115", "r139", "r151", "r152")):
        continue
    _obj = globals()[_name]
    if hasattr(_obj, "shape"):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict keys", list(_obj)[:24])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, "length", len(_obj))
    elif isinstance(_obj, (int, float, str, np.number)):
        print(_name, repr(_obj)[:120])
    else:
        print(_name, type(_obj).__name__)

print("=== INSTALLED TENSOR HASHES ===")
for _name in edited():
    _obj = load(_name)
    _a = _obj.detach().cpu().numpy() if hasattr(_obj, "detach") else np.asarray(_obj)
    print(_name, tuple(_a.shape),
          hashlib.sha256(np.asarray(_a, dtype=np.float32).tobytes()).hexdigest()[:12])

print("=== RECENT GATE-FIT NOTES ===")
_n = open("/work/notes.md").read()
for _heading in ["## Run 103", "## Run 104", "## Run 105", "## Run 106",
                 "## Run 107", "## Run 108", "## Run 109", "## Run 110"]:
    _start = _n.find(_heading)
    if _start >= 0:
        _end = _n.find("\n## ", _start + 1)
        print(_n[_start:(_end if _end >= 0 else len(_n))][:550])
```
