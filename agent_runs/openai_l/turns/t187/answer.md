The stored decoder samples come from training stays, and the record-derived drug-response kernels are still available. I’ll recover those kernels and the state-file layout, then obtain matching states from the safe edited model. Agreement on factual predictions plus identifiable dose-sensitive features would support a targeted readout; substantial mixing with unrelated features would weaken it.

```python
import numpy as np
from pathlib import Path

print("=== RECORD-DERIVED KERNELS ===")
for name in ["kernels19", "kernel_se19", "kernels22"]:
    obj = globals().get(name)
    print(name)
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(" ", key, np.asarray(value).reshape(-1)[:16].tolist())
    else:
        print(repr(obj)[:1000])

print("\n=== CONSOLIDATED CAUSAL NOTE ===")
for section in Path("/work/notes.md").read_text().split("\n## "):
    if section.startswith("Run 36"):
        print(section[:2200])

print("\n=== STATE LAYOUT ===")
for fname in ["/work/fwd_3/states_original.npz",
              "/work/fwd_4/states_original.npz"]:
    with np.load(fname) as data:
        print(fname)
        for key in data.files:
            arr = data[key]
            print(key, arr.shape, arr.dtype,
                  arr.reshape(-1)[:3].tolist() if arr.size else [])

for name in ["sx63", "pi73", "dtread73", "rate_state81",
             "cf_features81", "const_train67", "trans_train67",
             "centres_train67", "epsread73", "w63_cache"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "dict", list(obj)[:12])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    else:
        print(name, type(obj).__name__)

print("bundle_inputs73 globals:", bundle_inputs73.__code__.co_names)
print("bundle_inputs73 constants:", repr(bundle_inputs73.__code__.co_consts)[:700])

fit_state_stays172 = sorted(measure67["stay_id"].unique().tolist())
print("\nMATCHING TRAINING STAYS:", fit_state_stays172)
print("\nCURRENT EDITS:", edited())
```