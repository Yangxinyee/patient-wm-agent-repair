The current edit gives exactly zero for all four rollouts, and factual NLL is essentially unchanged on 200 tuning stays. That validates the edit’s behavior, not the causal hypothesis. I’ll now investigate the remaining norepinephrine signal; a reproducible late-hour effect after better adjustment would argue against complete dose invariance.

```python
import numpy as np, pandas as pd, torch, inspect
print("=== DATA VARIABLES ===")
for _k, _v in list(globals().items()):
    if not _k.startswith("_") and isinstance(_v, (np.ndarray, pd.DataFrame, pd.Series, torch.Tensor)):
        print(_k, type(_v).__name__, tuple(_v.shape))
print("=== DICTIONARIES ===")
for _k, _v in list(globals().items()):
    if not _k.startswith("_") and isinstance(_v, dict) and _k not in ("SHAPES", "VOCAB"):
        print(_k, len(_v), list(_v)[:10])
print("=== HELPER DEFINITIONS ===")
for _fn_name in ["pre_dose_features_v2", "crossfit_mean_v2",
                 "future_reading_v2", "orthogonal_coefficient_v2"]:
    _fn = globals()[_fn_name]
    try:
        print(inspect.getsource(_fn))
    except (OSError, TypeError):
        print(_fn_name, "names:", _fn.__code__.co_names,
              "variables:", _fn.__code__.co_varnames,
              "constants:", repr(_fn.__code__.co_consts)[:1600])
print("=== EDIT CONSTRUCTION NOTES ===")
for _p in open("/work/notes.md").read().split("\n\n"):
    if "embed." in _p or "flatten" in _p.lower() or "bin mean" in _p.lower():
        print(_p[:2200])
print("=== FORWARD FILES ===")
for _d in sorted(os.listdir("/work")):
    if _d.startswith("fwd_"):
        print(_d, os.listdir("/work/" + _d))
```