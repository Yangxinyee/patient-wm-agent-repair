```python
import os, json, inspect, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())
print("=== EARLY NOTEBOOK / IDENTIFICATION ===")
notes99 = open("/work/notes.md").read()
print(notes99[:11000])
print("=== RECENT KERNEL OBJECTS ===")
for name99 in sorted(list(globals())):
    if name99.endswith(("93", "94", "95", "96", "97", "98")) or name99 in (
        "idx80", "VOCAB", "NAMES", "SHAPES", "core", "df", "train", "tuning"
    ):
        obj99 = globals()[name99]
        if isinstance(obj99, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
            print(name99, type(obj99).__name__, tuple(obj99.shape))
        elif callable(obj99):
            print(name99, "callable", str(inspect.signature(obj99)) if inspect.isfunction(obj99) else "")
        elif isinstance(obj99, dict):
            print(name99, "dict", len(obj99), "keys:", list(obj99)[:18])
        elif isinstance(obj99, (list, tuple)):
            print(name99, type(obj99).__name__, len(obj99), repr(obj99[:3])[:350])
        else:
            print(name99, repr(obj99)[:500])
print("=== EDITS ===", edited())
with open("/work/notes.md", "a") as f99:
    f99.write("\n## Run 99: restart after label audit\n"
              "- Run 98 found 2250 incorrect old target bins; direct labels reduce but do not resolve the local/external NLL gap.\n"
              "- Recovering architecture and identical-reading diagnostics before any weight edit.\n"
              "- Python runs: 99; forward passes: 4; no submitted edits.\n")
```
