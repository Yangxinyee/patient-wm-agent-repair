The warm-up candidate failed: factual NLL rose by 0.01425 on the difficult tuning group, despite passing the local scan. I’ll restore the strength-2 fallback and locate the extra loss. If it is concentrated in a few readings, I’ll investigate decoder extrapolation; if it is widespread, the gate itself needs reconsideration.

```python
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Restore the validated fallback before any diagnostic can fail.
for _name, _tensor in r157_fallback.items():
    save_edit(_name, torch.as_tensor(_tensor))
r158_rejected = {k: v.copy() for k, v in r157_candidate.items()}

r158_o = pd.read_csv("/work/fwd_19/factual_original.csv")
r158_w = pd.read_csv("/work/fwd_19/factual_edited.csv")
r158_s = pd.read_csv("/work/fwd_18/factual_edited.csv")
print("CSV columns:", list(r158_o.columns))

r158_keys = [c for c in ["stay_id", "hour", "event"] if c in r158_o.columns]
assert all(r158_o[c].equals(r158_w[c]) and r158_o[c].equals(r158_s[c])
           for c in r158_keys)
r158_nll_cols = [c for c in r158_o.columns if "nll" in c.lower()]
assert len(r158_nll_cols) == 1, r158_nll_cols
r158_nll_col = r158_nll_cols[0]
r158_diag = r158_o[r158_keys].copy()
r158_diag["warm_loss"] = r158_w[r158_nll_col] - r158_o[r158_nll_col]
r158_diag["fallback_loss"] = r158_s[r158_nll_col] - r158_o[r158_nll_col]
r158_diag["extra_loss"] = r158_diag["warm_loss"] - r158_diag["fallback_loss"]
for _col in r158_o.columns:
    if "mean" in _col.lower() and np.issubdtype(r158_o[_col].dtype, np.number):
        r158_diag["warm_mean_shift"] = r158_w[_col] - r158_o[_col]
        r158_diag["fallback_mean_shift"] = r158_s[_col] - r158_o[_col]
        break
print("=== LARGEST EXTRA LOSSES ===")
print(r158_diag.nlargest(15, "extra_loss").to_string(index=False))
print("=== LOSS BY HOUR ===")
if "hour" in r158_diag:
    print(r158_diag.groupby("hour")[["warm_loss", "fallback_loss"]]
          .agg(["mean", "sum"]).round(5).to_string())
print("=== CACHED DECODER DISTRIBUTIONS ===")
for _event in np.unique(r144_events):
    _v = np.asarray(r144_D)[:, r144_events == _event]
    print(_event, np.quantile(_v, [0, .01, .1, .5, .9, .99, 1]).round(4))
print("=== RECONSTRUCTION CACHE ===")
for _name in sorted(list(globals())):
    if _name.startswith(("r124", "r129", "r137")):
        _v = globals()[_name]
        if hasattr(_v, "shape"):
            print(_name, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_name, "keys", list(_v)[:8])

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Forward 19 / Run 158: warm-up trial rejected\n"
        "On tuning 300..399, warm-up strength-2.6 NLL was 1.65674 vs "
        "1.64249 original (delta +0.01425). Local gate performance did not "
        "generalize. Restored the strength-2 insulin fallback before analysis.\n")
```
