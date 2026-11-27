The kernel still has both the rejected candidate and the original weights, allowing a direct audit without changing the model. The inspection stopped on an enumeration error. I’ll fix that and inspect the prior-to-head calculation; correctly sampled original predictions should remain consistent with the GPU results.

```python
import numpy as np
import torch
import pandas as pd
import inspect

print("=== HEAD IMPLEMENTATION ===")
for name, limit in [
    ("reconstruct_reading_head", 5000),
    ("corrected_register_head", 2400),
    ("latent_context_at", 1600),
    ("audit_head_logits", 1200),
]:
    obj = globals().get(name)
    print("\nFUNCTION", name)
    try:
        print(inspect.getsource(obj)[:limit])
    except Exception:
        print("globals:", obj.__code__.co_names)
        print("locals:", obj.__code__.co_varnames)
        print("constants:", [x for x in obj.__code__.co_consts
                             if isinstance(x, (str, int, float, tuple))][:45])

print("\n=== RECENT AUDIT VARIABLES ===")
snapshot142 = list(globals().items())
for k, v in snapshot142:
    if not (k.startswith("audit_") or k.startswith("r139") or k.startswith("r138")
            or k.startswith("r137") or k.startswith("r136")):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys()))
    elif isinstance(v, (str, float, int, tuple, list)):
        print(k, repr(v)[:180])

print("\n=== MOST RECENT ARRAY NAMES ===")
arrays142 = [(k, tuple(v.shape)) for k, v in snapshot142
             if not k.startswith("_")
             and isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series))]
print(arrays142[-45:])
```