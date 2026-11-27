The lag estimates suggest effects that persist and decay over several hours, not constant offsets. A value-head-only edit would leave other teacher-forced outputs unchanged, which makes it a useful first approach. I’ll check the remaining encoder discrepancy and recover the relevant feature arrays; a correct encoder should reproduce the exported posteriors numerically.

```python
import dis, numpy as np, torch
from pathlib import Path

print("=== ATOM COMPARISON ===")
for _code, _value in (("glucose_finger", 150.), ("glucose_lab", 150.),
                       ("map_arterial", 70.), ("map_cuff", 70.)):
    _a = rr_atom(_code, value=_value)
    _b = r95_correct_atom(_code, value=_value)
    _d = _a - _b
    print(_code, "fresh/cached-helper difference norm", float(_d.norm()),
          "max", float(_d.abs().max()))

print("\n=== KIND/UNIT CONSTRUCTION ===")
for _name in ("rr_atom", "r95_base_atom"):
    _fn = globals()[_name]
    _ins = list(dis.get_instructions(_fn))
    _keep = set()
    for _j, _i in enumerate(_ins):
        if _i.opname == "LOAD_CONST" and isinstance(_i.argval, str) and any(
                s in _i.argval for s in ("embed.kind", "embed.ekind", "categorical")):
            _keep.update(range(max(0, _j-2), min(len(_ins), _j+16)))
    print(_name)
    for _j in sorted(_keep):
        _i = _ins[_j]
        print(_i.opname, _i.argrepr)

print("\n=== VALIDATED CACHE INVENTORY ===")
for _n, _v in sorted(list(globals().items())):
    if _n.startswith(("r33_", "r79_", "r86_")):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_n, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_n, "dict", len(_v), list(_v)[:3])
        elif isinstance(_v, (list, tuple)):
            print(_n, type(_v).__name__, len(_v),
                  repr(_v[:2])[:160] if len(_v) else "")
        elif callable(_v) and hasattr(_v, "__code__"):
            print(_n, "function", _v.__code__.co_names)

print("\n=== CANDIDATE AND DYNAMICS OBJECTS ===")
for _n, _v in sorted(list(globals().items())):
    if any(s in _n.lower() for s in ("candidate", "trace", "kernel", "deconv", "innovation")):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_n, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_n, "dict", len(_v), list(_v)[:5])
        elif isinstance(_v, (list, tuple)):
            print(_n, type(_v).__name__, len(_v))
        elif callable(_v):
            print(_n, "function")

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 116\nPrior deconvolved kernels imply multi-hour, decaying responses. "
             "Prioritizing a value-head-only algebraic edit to preserve other teacher-forced outputs. "
             "Checking atom construction and validated cache dependencies; no weights edited. "
             "Counts: 116 Python runs, 6 forward passes.\n")
```
