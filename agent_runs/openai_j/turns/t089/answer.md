The centre-table mismatch is confirmed: bin 0 is “no value,” followed by 16 reading bins. The best local state reconstruction is close enough to investigate dose contrasts, though not to claim exact reproduction. I’ll locate the completed probe arrays and correct the mean calculation; this should reveal whether the value readout contains usable dose-sensitive features.

```python
import inspect
import dis
import numpy as np
import pandas as pd

print("=== REPAIR CACHE ===")
lines = []
for k, obj in list(globals().items()):
    if k.startswith("repair_") and not callable(obj):
        if hasattr(obj, "shape"):
            lines.append(f"{k}: {type(obj).__name__} {tuple(obj.shape)}")
        elif isinstance(obj, dict):
            lines.append(f"{k}: keys {list(obj.keys())[:12]}")
        elif isinstance(obj, (list, tuple)):
            lines.append(f"{k}: {type(obj).__name__} len={len(obj)}")
        else:
            lines.append(f"{k}: {str(obj)[:120]}")
print("\n".join(lines)[:3200])

print("\n=== LAST PROBE VARIABLES ===")
lines = []
for k, obj in list(globals().items())[-65:]:
    if hasattr(obj, "shape"):
        lines.append(f"{k}: {type(obj).__name__} {tuple(obj.shape)}")
    elif isinstance(obj, dict):
        lines.append(f"{k}: keys {list(obj.keys())[:8]}")
print("\n".join(lines)[:1300])

for fn_name, tail, budget in [
    ("repair_changed_action", True, 1100),
    ("repair_core_rows", True, 1500),
    ("reconstruct_reading_head", False, 1700),
]:
    fn_obj = globals()[fn_name]
    print("\n===", fn_name, "defaults", fn_obj.__defaults__, "===")
    print("locals:", fn_obj.__code__.co_varnames)
    try:
        text = inspect.getsource(fn_obj)
    except (OSError, TypeError):
        text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.Bytecode(fn_obj))
    print(text[-budget:] if tail else text[:budget])

def repair_centres17(meta):
    c = np.asarray(repair_centres_for_meta(meta), dtype=np.float64)
    if c.shape[1] == 16:
        c = np.column_stack([np.zeros(len(c)), c])
    assert c.shape == (len(meta), 17)
    return c

print("\nCorrected centre shape:", repair_centres17(head_meta).shape)
```