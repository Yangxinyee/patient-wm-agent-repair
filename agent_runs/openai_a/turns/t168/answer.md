The synthetic cases and factual-head features are still available. Before fitting a new decoder, I’ll check how the existing graft gates glucose and whether its protection features exclude reading-history and latent variation. A useful replacement must respond to dose contrasts without recreating the unstable feedback seen in the stronger trial.

```python
import re, inspect, numpy as np, pandas as pd
print("RELEVANT OBJECTS")
for key, obj in list(globals().items()):
    if key.startswith("_") or not re.search(
        r"project|protect|factual|synt|graft|feature_set|feat_sets|gate|checkpoint|ridge|sens|register|latent", key, re.I
    ):
        continue
    if isinstance(obj, np.ndarray):
        print(key, obj.shape)
    elif isinstance(obj, dict):
        print(key, "keys", list(obj.keys())[:9])
    elif isinstance(obj, (list, tuple)):
        print(key, type(obj).__name__, len(obj))
    elif isinstance(obj, (float, int, bool, str)):
        print(key, str(obj)[:130])
print("\nSYNTHETIC HEAD STRUCTURE")
for m0, obj in lab_head_results.items():
    print(m0, {k: (v.shape if hasattr(v, "shape") else type(v).__name__)
               for k, v in obj.items()})
print("\nHELPER DEFINITIONS")
for name0 in ["make_insulin_graft", "probe_head_eval", "repair_feature_sets"]:
    print("\n", name0)
    try:
        print(inspect.getsource(globals()[name0])[:2300])
    except (OSError, TypeError):
        compact_dis(globals()[name0], max_chars=2300)
```
