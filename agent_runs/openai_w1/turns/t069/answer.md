The later checks estimate a one-hour effect of about −1.92 per insulin unit and +3.33 per norepinephrine unit, with distinct multi-hour responses. These are conditional estimates, not proof that all confounding is absent. I’ll inspect the candidate residual channels and encoder helper next; the key test is whether they can change dose sensitivity while remaining nearly neutral on factual records.

```python
import inspect
import numpy as np
from pathlib import Path

print("=== REMAINING DYNAMICS / EDIT NOTES ===")
_s61 = Path("/work/notes.md").read_text()
_a61 = _s61.find("## Restart / Run 19")
_b61 = _s61.find("## Run 40")
print(_s61[_a61:_b61][:4600])

print("\n=== EXPECTED-DOSE FIT STRUCTURES ===")
for _drug61, _obj61 in rr51_assignment.items():
    print(_drug61)
    if isinstance(_obj61, dict):
        for _key61, _val61 in _obj61.items():
            if hasattr(_val61, "shape"):
                print(" ", _key61, type(_val61).__name__, tuple(_val61.shape))
            else:
                print(" ", _key61, repr(_val61)[:130])
    else:
        print(type(_obj61).__name__, repr(_obj61)[:350])

print("\n=== PHYSIOLOGICAL DIRECTIONS ===")
for _key61, _val61 in rr58_reading_dirs.items():
    if isinstance(_val61, dict):
        print(_key61, {
            k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:80])
            for k, v in _val61.items()
        })
    elif hasattr(_val61, "shape"):
        print(_key61, tuple(_val61.shape), np.asarray(_val61).reshape(-1)[:8])
    else:
        print(_key61, type(_val61).__name__, repr(_val61)[:300])

print("\n=== ENCODER HELPER ===")
try:
    print(inspect.getsource(rr58_observation_encoding)[:3500])
except (OSError, TypeError):
    print("arguments:", rr58_observation_encoding.__code__.co_varnames)
    print("globals:", rr58_observation_encoding.__code__.co_names)
    print("constants:", repr(rr58_observation_encoding.__code__.co_consts)[:2000])

print("\n=== OTHER RETAINED FIT / CACHE OBJECTS ===")
for _nm61 in sorted(list(globals())):
    if _nm61.startswith(("rr", "r4")) and not _nm61.startswith(
            ("rr50", "rr51", "rr56", "rr57", "rr58", "rr59", "rr60")):
        _v61 = globals()[_nm61]
        if callable(_v61):
            print(_nm61, "callable")
        elif isinstance(_v61, dict):
            print(_nm61, "dict", list(_v61)[:5])
        elif hasattr(_v61, "shape"):
            print(_nm61, tuple(_v61.shape))
        elif isinstance(_v61, (list, tuple)):
            print(_nm61, type(_v61).__name__, len(_v61))
print("reserved posterior neurons:", rr57_reserved)
print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f61:
    _f61.write("\n## Run 61\n"
               "Recovered conditional one-hour effects (-1.9202 insulin, +3.3339 NE) and "
               "six-hour deconvolved kernels. Inspecting expected-dose fits and observation helper "
               "before a residual-based posterior edit. No edits. "
               "Counts: 61 Python runs, 6 forward passes.\n")
```