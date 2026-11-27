The architecture confirms that dose information reaches readings through a nonlinear state update. The stronger edit passed factual validation but did not improve doubling, so simply increasing its strength is not justified. I’ll inspect the earlier causal evidence and hour-by-hour rollout shifts; that should reveal whether the correction fades or reverses during rollout.

```python
import numpy as np
import pandas as pd
import inspect
from pathlib import Path

rr_notes = Path("/work/notes.md").read_text()
print("=== IDENTIFICATION NOTES, CONTINUED ===")
print(rr_notes[4000:9500])

print("\n=== GRAFT / DECODER OBJECTS ===")
for rr_name, rr_obj in sorted(list(globals().items())):
    if rr_name.startswith("rr_"):
        continue
    if any(s in rr_name.lower() for s in
           ["graft", "robust", "decoder", "gate", "checkpoint"]):
        if hasattr(rr_obj, "shape"):
            print(rr_name, type(rr_obj).__name__, tuple(rr_obj.shape))
        elif isinstance(rr_obj, dict):
            print(rr_name, "keys:", list(rr_obj.keys())[:15])
        elif callable(rr_obj):
            try:
                print(rr_name, inspect.signature(rr_obj))
            except Exception:
                print(rr_name, "callable")
        elif isinstance(rr_obj, (int, float, str, list, tuple)):
            print(rr_name, repr(rr_obj)[:180])

print("\n=== ROLLOUT FILE STRUCTURE ===")
for rr_k in (9, 11):
    for rr_model in ("original", "edited"):
        rr_path = Path(f"/work/fwd_{rr_k}/dose_edit_{rr_model}.csv")
        if rr_path.exists():
            rr_df = pd.read_csv(rr_path)
            print(rr_k, rr_model, rr_df.shape, list(rr_df.columns))
            print(rr_df.head(2).to_string(index=False))
            if {"edit", "hour"}.issubset(rr_df.columns):
                rr_nums = [
                    c for c in rr_df.select_dtypes(include="number").columns
                    if c not in ("stay_id", "hour")
                ]
                print(rr_df.groupby(["edit", "hour"])[rr_nums].mean()
                      .round(3).to_string())

print("\n=== RETAINED CPU FUNCTIONS ===")
for rr_name, rr_obj in sorted(list(globals().items())):
    if callable(rr_obj) and not rr_name.startswith("rr_"):
        if any(s in rr_name.lower() for s in
               ["head", "input", "step", "embed", "encode", "feature", "graft"]):
            try:
                print(rr_name, inspect.signature(rr_obj))
            except Exception:
                pass
```