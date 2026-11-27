The insulin and MAP corrections use separate neuron pairs, so their strengths can be adjusted independently. A nonlinear MAP correction might also improve doubling more efficiently than another linear gain increase. I’ll check the cached counterfactual features and previous fit results first; held-out factual loss will determine whether that approach is viable.

```python
import inspect
from pathlib import Path
import numpy as np
import pandas as pd

print("COUNTERFACTUAL CACHE STRUCTURE")
print("wm_cf146 keys:", list(wm_cf146.keys()))
for audit230_key, audit230_val in wm_cf146.items():
    print(audit230_key, type(audit230_val).__name__)
    if isinstance(audit230_val, dict):
        print("  keys:", list(audit230_val.keys())[:12])
        for audit230_subkey, audit230_subval in list(audit230_val.items())[:3]:
            if isinstance(audit230_subval, dict):
                print(" ", audit230_subkey, {
                    k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else "")
                    for k, v in audit230_subval.items()
                    if k in ("inputs", "hidden", "logits", "active", "delta_rate", "mean")
                })

print("\nMETADATA TABLES")
for audit230_name, audit230_val in list(globals().items()):
    if (isinstance(audit230_val, pd.DataFrame)
        and len(audit230_val) in (4510, 4751)
        and "event" in audit230_val.columns):
        print(audit230_name, audit230_val.columns.tolist()[:18])

print("\nMOMENT/FISHER HELPERS")
for audit230_fn in (response_design208, factual_curvature218, scalar_eval208):
    print(audit230_fn.__name__, inspect.signature(audit230_fn))
    print("globals:", audit230_fn.__code__.co_names)
    print("fields:", [c for c in audit230_fn.__code__.co_consts if isinstance(c, str)])

print("\nFIT MASKS")
for audit230_name in ("fitall208", "cvall208", "mapmask208", "tunemap208",
                      "events206", "moment218_A", "moment218_b"):
    if audit230_name in globals():
        audit230_val = globals()[audit230_name]
        print(audit230_name, getattr(audit230_val, "shape", ""),
              int(np.asarray(audit230_val).sum())
              if np.asarray(audit230_val).dtype == bool else "")

print("\nPREVIOUS HEAD-CONTRAST FITS")
audit230_table = pd.read_csv("/work/head_contrast_cv.csv")
print("columns:", audit230_table.columns.tolist())
print(audit230_table.head(6).to_string(index=False))

audit230_notes = Path("/work/notes.md").read_text()
audit230_pos = audit230_notes.find("## Run 218")
if audit230_pos >= 0:
    print("\nPREVIOUS MOMENT ATTEMPT")
    print(audit230_notes[audit230_pos:audit230_pos + 1800])

print("\nINSULIN NEURONS:", trial158_free,
      "MAP NEURONS:", head177_free)
```
