The private dose coordinates now nearly match the edited GPU states, although the ordinary state still does not. That is encouraging but does not validate the whole replay. I’ll inspect the isolated action path and its dose decoder next. A roughly doubled or halved transported signal would shift suspicion toward the value head or rollout feedback.

```python
import inspect
import numpy as np
import torch

print("=== ACTION/REPLAY HELPER SIGNATURES ===")
for name in ["action_vectors4", "exact_u162", "wm_set",
             "action_atom_exact", "head_features_layout", "correction_data",
             "mechanistic_shifts", "response_pair68"]:
    fn = globals().get(name)
    if fn is not None:
        print(name, inspect.signature(fn),
              "weight globals:",
              [k for k in fn.__code__.co_names
               if k in globals() and isinstance(globals()[k], dict)])

print("=== CAPTURED CURRENT ACTION STEP ===")
pred188, loc188 = capture177(wm_step_boundary176, 31)
for key, val in loc188.items():
    if isinstance(val, (np.ndarray, torch.Tensor)):
        print(key, tuple(val.shape))
    elif isinstance(val, (str, int, float, bool, type(None))):
        print(key, repr(val))
    elif isinstance(val, (list, tuple, dict)):
        print(key, type(val).__name__, len(val))

sid188 = int(s4["stay_id"][31])
minute188 = int(round(float(s4["hour"][31]) * 60))
grp188 = groups4[(sid188, minute188)]
print("group type:", type(grp188).__name__)
print(str(grp188)[:1800])
print("private coordinates:", idx182.tolist())
for j in [3, 4, 7, 31, 32]:
    hh = np.asarray(s4["x"][j, 192:960]).reshape(4, 192)
    print("bundle", j, "hour", float(s4["hour"][j]),
          "h private mean", hh[:, idx182].mean(0).round(7).tolist(),
          "out private", np.asarray(s4["x"][j, idx182]).round(7).tolist())

print("=== TRANSPORT / HEAD OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or inspect.isfunction(obj):
        continue
    if any(k in name.lower() for k in
           ("strength", "pair", "decode", "center", "gate", "tilt", "channel",
            "beta", "phi", "causal")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:8])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, str(obj)[:220] if len(obj) < 12 else len(obj))
        elif isinstance(obj, (int, float, str)):
            print(name, repr(obj)[:100])
```
