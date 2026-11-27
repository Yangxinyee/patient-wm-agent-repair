The cached probe has the right components for testing direct action transmission. Its immediate failure likely comes from multiplying 17 bin probabilities by only 16 reading-bin centres. I’ll inspect the helper and cached arrays before correcting that alignment. If the repaired probe shows weak local dose transmission, it would support testing a targeted recurrent-path edit rather than another input gain.

```python
import inspect
import dis
import numpy as np
import pandas as pd

print("=== MOST RECENT CACHED OBJECTS ===")
lines = []
for k, obj in list(globals().items())[-140:]:
    if k.startswith("_"):
        continue
    if hasattr(obj, "shape"):
        lines.append(f"{k}: {type(obj).__name__} {tuple(obj.shape)}")
    elif isinstance(obj, dict):
        lines.append(f"{k}: dict keys={list(obj.keys())[:12]}")
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{k}: {type(obj).__name__} len={len(obj)}")
    elif isinstance(obj, (int, float, str, bool)):
        lines.append(f"{k}: {str(obj)[:90]}")
print("\n".join(lines)[:3900])

for fn_name, budget in [
    ("repair_centres_for_meta", 700),
    ("repair_core_rows", 1900),
    ("repair_changed_action", 1900),
]:
    fn_obj = globals()[fn_name]
    print("\n=== HELPER:", fn_name, "===")
    try:
        text = inspect.getsource(fn_obj)
    except (OSError, TypeError):
        text = "\n".join(
            f"{i.opname} {i.argrepr}" for i in dis.Bytecode(fn_obj)
        )
    print(text[:budget])

print("\n=== CORE RECONSTRUCTION CHECKS ===")
if "core_check_table" in globals():
    print(core_check_table.sort_values("out_rmse").head(4).to_string(index=False))
print("VALUE CENTRE LENGTHS:",
      {k: len(v) for k, v in VOCAB["value_bin_centres"].items()})
```