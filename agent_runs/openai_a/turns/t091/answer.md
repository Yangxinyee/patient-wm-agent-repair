The stronger insulin edit raised factual NLL by 0.00170, within the 0.005 limit on these 100 stays. It remains a candidate, but its dose response is untested. I’ll recover the construction and causal estimates before testing rollouts; improvement toward the record-derived effects, with preserved factual fit, would support keeping it.

```python
import os, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())
_recovery_notes = open("/work/notes.md").read()
print("=== EARLY NOTEBOOK ===")
print(_recovery_notes[:14000])
print("=== NOTEBOOK HEADINGS ===")
print("\n".join(x for x in _recovery_notes.splitlines() if x.startswith("#")))
print("=== RETAINED VARIABLES ===")
for _name in sorted(list(globals())):
    if _name.startswith("_") or _name in {"In", "Out", "get_ipython", "exit", "quit"}:
        continue
    _obj = globals()[_name]
    if callable(_obj):
        print(_name, "callable")
    elif isinstance(_obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", list(_obj.keys())[:16])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj), repr(_obj[:5])[:240])
    elif isinstance(_obj, (str, int, float, bool, type(None))):
        print(_name, repr(_obj)[:240])
print("=== CURRENT EDITS ===", edited())
print("=== TUNING IDS ===", list(tuning_ids[:100]))
```
