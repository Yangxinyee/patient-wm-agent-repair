The prior GPU files confirm that the strong edit changed generated readings substantially even though its teacher-forced NLL passed. That could explain why increasing its strength did not improve the treatment readout. I’ll compare the hourly responses and locate the newer protected candidates. More stable responses across later hours would support trying one of those instead.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== HOURLY INSULIN SHIFTS: RETAINED AND STRONG EDITS ===")
for k, label in [(9, "beta=-2"), (11, "beta=-24")]:
    tables = {}
    for model in ["original", "edited"]:
        t = pd.read_csv(f"/work/fwd_{k}/dose_edit_{model}.csv")
        eligible = t[t["reached"] & t["edit"].str.startswith("insulin")]
        tables[model] = eligible.pivot_table(index="hour", columns="edit", values="shift", aggfunc="mean")
        print(label, model, "eligible readout",
              eligible.groupby("edit")["shift"].mean().round(4).to_dict())
    print(pd.concat(tables, axis=1).round(3).to_string())
    a = pd.read_csv(f"/work/fwd_{k}/dose_edit_original.csv")
    b = pd.read_csv(f"/work/fwd_{k}/dose_edit_edited.csv")
    joined = a.merge(b, on=["stay_id", "hour", "edit"], suffixes=("_original", "_edited"))
    joined["factual_difference"] = joined["factual_edited"] - joined["factual_original"]
    print("generated factual mean difference:",
          joined.groupby("edit")["factual_difference"].agg(["mean", "std"]).round(3).to_string())

print("=== NEWER CANDIDATE CACHES / HEAD CONSTRUCTION ===")
for key in ["r2_candidates", "r3_candidates", "scaled_candidates3", "scaled_cpu_checks3",
            "new_decoder_candidates", "input_tensors3", "rank_template",
            "gate_w", "gate_b", "graft_removed_mean"]:
    if key not in globals():
        continue
    v = globals()[key]
    if isinstance(v, dict):
        print(key, "keys:", list(v.keys())[:22])
        if v:
            kk = next(iter(v))
            vv = v[kk]
            print("first entry:", kk, type(vv).__name__,
                  "keys=" + repr(list(vv.keys())) if isinstance(vv, dict)
                  else "shape=" + repr(getattr(vv, "shape", None)))
    elif hasattr(v, "shape"):
        print(key, "shape", v.shape, "norm", float(np.linalg.norm(np.asarray(v))),
              "range", float(np.min(np.asarray(v))), float(np.max(np.asarray(v))))
    else:
        print(key, repr(v)[:450])

for key in ["head_average_probs3", "graft_probs", "sc_cache_inputs"]:
    if key in globals():
        fn = globals()[key]
        print(key, "references", fn.__code__.co_names,
              "locals", fn.__code__.co_varnames,
              "constants", repr(fn.__code__.co_consts)[:550])

print("=== LATEST CONTRAST VARIABLE NAMES ===")
print(sorted(k for k in globals()
             if k.startswith(("r2_", "r3_", "sc_", "gc_", "ic_"))))
```
