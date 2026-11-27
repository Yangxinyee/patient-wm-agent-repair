The architecture confirms that doses enter both the latent posterior and the recurrent update; neither pathway has been repaired yet. I’ll identify the existing reconstruction helpers so I can test these pathways directly. The hypothesis is that a targeted state correction will preserve factual fit better than broad dose amplification.

```python
import inspect, numpy as np, pandas as pd, torch
print("--- FUNCTIONS ---")
for k, v in list(globals().items()):
    if not k.startswith("_") and inspect.isfunction(v):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, sig, "references:", v.__code__.co_names[:18])
print("\n--- LARGE ARRAYS / DATAFRAMES ---")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)) and v.size if False else False:
        pass
    if isinstance(v, np.ndarray) and v.size > 100:
        print(k, v.shape, v.dtype)
    elif isinstance(v, torch.Tensor) and v.numel() > 100:
        print(k, tuple(v.shape), str(v.dtype))
    elif isinstance(v, pd.DataFrame) and len(v) > 100:
        print(k, v.shape, list(v.columns)[:10])
print("\n--- DICTIONARIES ---")
for k, v in list(globals().items()):
    if not k.startswith("_") and isinstance(v, dict) and len(v) > 3:
        print(k, len(v), list(v)[:6])
print("\nCurrent edits:", edited())
```
