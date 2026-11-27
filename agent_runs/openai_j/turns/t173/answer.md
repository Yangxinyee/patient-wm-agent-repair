The cached Monte Carlo features are available; the inventory error did not affect them or change any weights. I’ll check the fit’s scaling and dose-contrast targets before installing anything. The candidate must reproduce its predicted logits exactly—otherwise its local factual-safety estimate is not trustworthy.

```python
import dis
import numpy as np
import torch

print("=== HEAD AND FEATURE SCALING ===")
print("axis:", repair_axis)
print("logit direction:", wm_v151)
print("input scale range:", np.min(wm_s152), np.max(wm_s152))
print("gate norm/bias:", np.linalg.norm(wm_gate153[:-1]), wm_gate153[-1])
print("original value-head shapes:",
      {k: tuple(v.shape) for k, v in wm_weights145.items()})
for _r157_name in ("wm_train145", "wm_tune145"):
    _r157_obj = globals()[_r157_name]
    print(_r157_name, {
        k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        for k, v in _r157_obj.items()
    })
    print("raw fields:", {
        k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        for k, v in _r157_obj["raw"].items()
    })

print("\n=== FIT AND COUNTERFACTUAL STRUCTURE ===")
for _r157_name, _r157_key in (
    ("wm_fits152", (1.0, 0.0001)),
    ("wm_fits153", (1.0, 0.3)),
    ("wm_cf146", ("insulin", 2.0)),
    ("wm_tau_targets151", ("insulin", 2.0)),
):
    _r157_v = globals()[_r157_name][_r157_key]
    print(_r157_name, _r157_key, type(_r157_v).__name__)
    if isinstance(_r157_v, dict):
        for _r157_k, _r157_a in _r157_v.items():
            print(" ", _r157_k, tuple(_r157_a.shape)
                  if hasattr(_r157_a, "shape") else type(_r157_a).__name__)
    elif hasattr(_r157_v, "shape"):
        print(" shape", tuple(_r157_v.shape),
              "first", np.asarray(_r157_v).reshape(-1)[:8])
    elif isinstance(_r157_v, (tuple, list)):
        print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else "")
               for v in _r157_v])
print("\nmetadata:", head_meta.columns.tolist())
print(head_meta.head(2).to_string(index=False))
print("\n=== RAW LOSS IMPLEMENTATION ===")
dis.dis(wm_rawloss152)
print("\n=== MOMENT IMPLEMENTATION ===")
dis.dis(wm_moments151)
print("\nActive edits:", edited())
```
