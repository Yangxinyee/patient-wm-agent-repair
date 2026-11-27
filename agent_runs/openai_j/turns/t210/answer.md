The frontier shows that stronger linear MAP tilts quickly exhaust the factual-loss budget, while doubled-dose sensitivity remains weak. I’ll test whether a record-fitted calibration can remove avoidable factual error before adding more response. Improvement on both held-out training stays and tuning stays would support this approach; training-only improvement would not.

```python
import numpy as np
import torch
import inspect

print("TILT STATISTIC")
print("wm_v151:", wm_v151)
print("head177_coef norm:", np.linalg.norm(head177_coef))
print("MAP matrix scales:")
for name189 in ["map164_K", "map164_C_ne", "map164_C_i"]:
    a189 = np.asarray(globals()[name189])
    print(name189, "norm", np.linalg.norm(a189),
          "diag quantiles", np.quantile(np.diag(a189), [0, .1, .5, .9, 1]))
print("map164_scale quantiles:", np.quantile(map164_scale, [0, .1, .5, .9, 1]))

print("\nTRAIN SPLIT MASKS AND METADATA")
for name189, val189 in sorted(list(globals().items())):
    if isinstance(val189, np.ndarray) and val189.ndim == 1 and len(val189) == 4510:
        if val189.dtype == bool or "mask" in name189 or "fit" in name189 or "cv" in name189:
            print(name189, val189.dtype, "sum", val189.sum())
    elif isinstance(val189, torch.Tensor) and val189.ndim == 1 and len(val189) == 4510:
        if val189.dtype == torch.bool:
            print(name189, val189.dtype, "sum", int(val189.sum()))
    elif hasattr(val189, "columns") and "stay_id" in val189.columns and len(val189) == 4510:
        print(name189, val189.shape, val189.columns.tolist())

print("\nLOCAL MONTE CARLO OBJECTS")
for name189, val189 in sorted(list(globals().items())):
    if isinstance(val189, dict) and "inputs" in val189 and "logits" in val189:
        inp189 = val189["inputs"]
        print(name189, "inputs", getattr(inp189, "shape", None))
    elif isinstance(val189, dict) and val189:
        sample189 = next(iter(val189.values()))
        if isinstance(sample189, dict) and "inputs" in sample189 and "logits" in sample189:
            print(name189, "nested keys", list(val189.keys())[:12])

print("\nFITTING HELPERS")
for name189 in ["wm_loss151", "wm_moments151", "wm_probs151", "map164_mean"]:
    fn189 = globals()[name189]
    print(name189, inspect.signature(fn189),
          "globals", fn189.__code__.co_names)

print("\nPREVIOUS CALIBRATION NOTES")
for sec189 in sections187:
    if re.search(r"calibrat|bias", sec189.split("\n", 1)[0], re.I):
        print(sec189[:1200])
```
