Across all 400 stays, factual NLL increased by only 0.000082 nats; the stay-clustered 95% interval ends at 0.000267, well below the limit. I’ll leave the weights unchanged and prepare a window-average MAP check. A persistent adjusted norepinephrine association across hours 13–24 would challenge the null edit; an isolated endpoint association would be weaker evidence.

```python
import inspect
import numpy as np

print("=== RECORD ARRAYS ===")
for key in ["arr", "aa", "ids", "values", "rates", "n_stays", "n_times",
            "hours_all", "groups_all", "treatment_hours_v2", "stay_folds_v2",
            "row_folds_v2", "map_baseline_v3"]:
    ob = globals().get(key)
    print(key, type(ob).__name__, getattr(ob, "shape", None),
          str(ob)[:160] if np.isscalar(ob) else "")
    if isinstance(ob, dict):
        print(" keys:", list(ob)[:12])
print("events:", events)
print("hour range:", np.nanmin(hours_all), np.nanmax(hours_all))
print("treatment-hour range:",
      np.nanmin(treatment_hours_v2), np.nanmax(treatment_hours_v2))
print("NE features:", ne_features_v2)

print("=== EXISTING ANALYSIS FUNCTIONS ===")
for key in ["future_reading_v2", "matched_crossfit_v3", "orthogonal_coefficient_v2"]:
    fn = globals()[key]
    print(key, inspect.signature(fn))
    try:
        src = inspect.getsource(fn)
        print(src[:2200])
    except (OSError, TypeError):
        print("Source unavailable; argument/local names:", fn.__code__.co_varnames)
        print("Constants:", str(fn.__code__.co_consts)[:1200])

print("=== STORED LATE RESIDUAL STRUCTURE ===")
for lag, ob in ne_late_residuals_v3.items():
    print("lag:", lag, "type:", type(ob).__name__)
    if isinstance(ob, dict):
        for key, value in ob.items():
            print(" ", key, type(value).__name__, getattr(value, "shape", None))
    elif isinstance(ob, (tuple, list)):
        print([(type(v).__name__, getattr(v, "shape", None)) for v in ob])
print("=== NUISANCE SETTINGS ===")
print("settings_v3:", settings_v3)
print("nuisance_settings_v2:", nuisance_settings_v2)
```