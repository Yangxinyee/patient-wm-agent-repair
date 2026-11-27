The latest pass exported internal states for the selected training stays; it did not test an edit or establish a treatment effect. The record-based estimates suggest effects in both pathways, but we still need a reliable way to change them without hurting factual fit. I’ll inspect the architecture and retained variables before choosing an algebraic edit; a usable edit must respond to dose changes while minimally disturbing recorded histories.

```python
import os, json, inspect, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())
print("=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])
print("=== VOCAB ===")
print(json.dumps(VOCAB, default=str)[:18000])
print("=== PERSISTENT VARIABLE INVENTORY ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(value, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(key, type(value).__name__, tuple(value.shape))
    elif callable(value) and getattr(value, "__module__", "") in {"__main__", None}:
        code = getattr(value, "__code__", None)
        print(key, "function", code.co_varnames[:code.co_argcount] if code else "")
    elif isinstance(value, dict):
        print(key, "dict", list(value.keys())[:18])
    elif isinstance(value, (list, tuple)):
        print(key, type(value).__name__, "len", len(value), repr(value[:4])[:220])
    elif isinstance(value, (str, int, float, bool)):
        print(key, repr(value)[:160])
print("=== EARLY NOTEBOOK ===")
print(open("/work/notes.md").read()[:9000])
print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Run 19\n"
            "Forward 4 exported 5,215 original bundle states on representative training stays. "
            "No weights edited. Rechecking architecture and persistent analysis objects before "
            "a directly validated algebraic correction; treatment conclusions remain conditional "
            "on pre-dose-history adjustment.\n"
            "Counts: 19 Python runs, 4 forward passes.\n")
```
