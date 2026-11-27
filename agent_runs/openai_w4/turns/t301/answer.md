Increasing the insulin gain barely improved doubling and weakened the halving response. That argues against simple gain scaling as the fix. I’ll inspect the retained channel construction and replay helpers to separate the dose-response path from factual compensation; a selective edit should improve rollout sensitivity without spending much more NLL budget.

```python
from pathlib import Path
import inspect, numpy as np, torch

with open("/work/notes.md", "a") as f:
    f.write("\n## GPU passes 27–28; Run 260\n"
            "Gain-1.5/gate-28 passed GPU factual on 98 stays: delta +.00327. "
            "Reached-row rollout shifts: insulin x2 -7.777, half +2.207; "
            "NE x2 +.748, half -3.178. Increasing insulin gain from 1.25 "
            "did not proportionally increase response and worsened halving. "
            "Inspecting retained response/compensation construction before further edits.\n")

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRELEVANT RETAINED VARIABLES")
keys260 = [k for k in list(globals()) if
           any(s in k.lower() for s in
               ("258", "259", "250", "192", "186", "180", "repair",
                "reference_bank", "replay_factual", "gate_fits",
                "capture_transition", "transition73"))]
for k in sorted(keys260):
    v = globals()[k]
    if callable(v):
        try:
            print(k, "FUNCTION", inspect.signature(v),
                  "names", v.__code__.co_names)
        except Exception:
            print(k, "CALLABLE")
    elif isinstance(v, dict):
        print(k, "DICT", len(v), "keys", list(v)[:18])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (int, float, str, tuple, list)):
        s = repr(v)
        print(k, s[:350])
    else:
        print(k, type(v).__name__)
print("\nCANDIDATE STRUCTURE")
for k in ("r258_candidates", "r259_backup", "r250_passing",
          "repair_bank", "reference_bank125"):
    v = globals().get(k)
    if isinstance(v, dict):
        for kk, vv in list(v.items())[:4]:
            print(k, repr(kk), type(vv).__name__,
                  list(vv)[:14] if isinstance(vv, dict)
                  else tuple(vv.shape) if hasattr(vv, "shape")
                  else repr(vv)[:180])
```
