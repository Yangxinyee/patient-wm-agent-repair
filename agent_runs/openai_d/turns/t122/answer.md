The history-adjusted insulin slope is stable across dose and hour subsets, supporting the correction’s direction. The cumulative rollout targets still depend on estimated delayed effects, so they remain provisional. I’ll inspect the cached head calculation and test stronger corrections locally first. If their factual cost exceeds the budget, amplification alone is not viable.

```python
import inspect
import re
import numpy as np
import torch

print("=== EXACT LOCAL HELPER ===")
for name in ["local104", "trial_gelu_np", "rh_eval", "rh_register"]:
    obj = globals().get(name)
    if callable(obj):
        print(name, "arguments:", inspect.signature(obj))
        print("referenced names:", obj.__code__.co_names)
        try:
            print(inspect.getsource(obj)[:1800])
        except (OSError, TypeError):
            print("Source unavailable; function remains callable.")

print("=== RECENT VARIABLES (NUMERIC SUFFIXES) ===")
lines109 = []
for name, obj in sorted(list(globals().items())):
    if re.search(r"(99|100|101|102|103|104|105|106)(?:_|$)", name):
        if isinstance(obj, dict):
            desc = "dict " + str(list(obj.keys())[:18])
        elif hasattr(obj, "shape"):
            desc = type(obj).__name__ + " " + str(tuple(obj.shape))
        elif isinstance(obj, (int, float, str, list, tuple)):
            desc = str(obj)[:180]
        elif callable(obj):
            desc = "callable"
        else:
            desc = type(obj).__name__
        lines109.append(name + ": " + desc)
print("\n".join(lines109)[:5000])

print("=== ORIGINAL FACTUAL CSV SCHEMA ===")
import pandas as pd
print(pd.read_csv("/work/fwd_10/factual_original.csv", nrows=2).to_string(index=False))
```