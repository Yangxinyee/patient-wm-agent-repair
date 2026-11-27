The head-only test failed: meaningful sensitivity gains caused far more than 0.005 nats of factual deterioration. I’ll leave those candidates unsaved and investigate the action-dependent posterior instead. If it retains dose innovations more clearly than the value head does, we may be able to correct unexpected dose changes without shifting ordinary factual predictions much.

```python
import sys
import numpy as np
import torch

print("=== LABEL FUNCTION METADATA ===")
for name in ["insulin_labels_fixed", "build_history_features",
             "rp_mlp", "pre_context_repair"]:
    fn = globals()[name]
    print(name, "variables:", fn.__code__.co_varnames)
    print(" references:", fn.__code__.co_names)
    print(" constants:", [x for x in fn.__code__.co_consts
                         if isinstance(x, (str, int, float, tuple))
                         and len(str(x)) < 220])

print("\n=== HISTORY / NUISANCE OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") and name not in ["_nuisance", "_moment_inputs"]:
        continue
    if isinstance(obj, dict) and any(s in name.lower() for s in
            ["hist", "nuis", "orth", "causal", "resid", "fit", "label"]):
        print(name, "keys:", list(obj)[:12])
        for key, val in list(obj.items())[:4]:
            print(" ", key, type(val).__name__,
                  tuple(val.shape) if hasattr(val, "shape") else
                  (list(val)[:10] if isinstance(val, dict) else ""))
    elif isinstance(obj, np.ndarray) and obj.ndim == 1 and any(
            s in name.lower() for s in ["eps", "resid", "innov", "label"]):
        print(name, obj.shape)

# Capture the already validated forward-only transition's intermediate arrays.
capture81 = {}
def profile81(frame, event, arg):
    if event == "return" and frame.f_code is transition73.__code__:
        capture81.update(frame.f_locals.copy())

ix81 = np.flatnonzero(
    (np.asarray(repair_bank["has_insulin"]) |
     np.asarray(repair_bank["has_norepinephrine"])) &
    np.isin(np.arange(len(repair_bank["x"])), valid73)
)[:24]
old_profile81 = sys.getprofile()
try:
    sys.setprofile(profile81)
    transition73(ix81)
finally:
    sys.setprofile(old_profile81)

print("\n=== TRANSITION INTERMEDIATES ===")
for name, obj in capture81.items():
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, list):
        print(name, "list", len(obj))

print("\n=== SMALL RECORD SAMPLE ===")
print(repair_records.loc[repair_records.event.isin(
    ["insulin", "norepinephrine"]),
    ["stay_id", "hour", "event", "rate", "bundle_index"]].head(12).to_string(index=False))
print("Panel index:", pan.index.names)
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 81\n"
        "Head-only rank-one calibration cannot restore direct effects within "
        "the factual budget: meaningful increases cost far above 0.005 nats. "
        "No candidates saved. Investigating posterior action innovations and "
        "a targeted transition repair instead of globally tilting the value head.\n"
    )
```
