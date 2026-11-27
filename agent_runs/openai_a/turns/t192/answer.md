The later threshold analysis supports an immediate insulin effect near −1.93 and a sustained doubling target near −11 glucose units. The retained edit reaches only −2.38; making it much stronger previously failed to help. I’ll inspect the hourly rollouts and candidate construction to distinguish weak dose decoding from feedback cancellation. An early improvement that disappears later would support the feedback explanation.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== PRIOR GPU OUTPUT SCHEMAS ===")
for k in [8, 9, 10, 11]:
    folder = Path(f"/work/fwd_{k}")
    for path in sorted(folder.glob("*.csv")):
        tab = pd.read_csv(path)
        print(path.name, "request", k, "shape", tab.shape)
        print("columns:", list(tab.columns))
        print(tab.head(2).to_string(index=False))

print("=== RELEVANT FUNCTIONS ===")
for key, val in sorted(list(globals().items())):
    if inspect.isfunction(val) and any(s in key.lower() for s in
       ["head", "graft", "contrast", "causal", "candidate", "check", "input", "rec", "state"]):
        try:
            print(key, inspect.signature(val))
        except Exception:
            print(key)

print("=== CONSTRUCTION SOURCE, IF AVAILABLE ===")
for key in ["make_insulin_graft", "cpu_head_parts", "matched_causal_targets163"]:
    if key in globals():
        fn = globals()[key]
        try:
            src = inspect.getsource(fn)
            print(key, src[:1900])
        except Exception:
            print(key, "global references:", fn.__code__.co_names)
            print("arguments/local names:", fn.__code__.co_varnames[:35])
            print("constants:", repr(fn.__code__.co_consts)[:1000])

print("=== SAVED CANDIDATE STRUCTURES ===")
for key in ["rm_candidate_cache", "graft_checkpoint_beta2", "graft_checkpoint_beta24",
            "protected_input_sets", "decode_feature_sets", "robust_candidates3",
            "sc_new_candidates", "graft_candidate", "candidate3"]:
    if key not in globals():
        continue
    obj = globals()[key]
    if isinstance(obj, dict):
        print(key, "keys", list(obj.keys())[:18])
        for kk, vv in list(obj.items())[:3]:
            print(" ", kk, type(vv).__name__, getattr(vv, "shape", ""),
                  repr(vv)[:150] if not hasattr(vv, "shape") else "")
    else:
        print(key, type(obj).__name__, getattr(obj, "shape", ""), repr(obj)[:180])
```
