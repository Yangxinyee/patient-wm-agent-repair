The retained helpers include the validated atom and set encoders, but the broad inventory was again too long. I’ll inspect only those helpers, the transition tensors, and the saved-state layout. This should give us a precise next-step reconstruction; disagreement with the saved states would mean its synthetic contrasts are not trustworthy.

```python
import inspect, json
from pathlib import Path

print("=== RECONSTRUCTION HELPER DETAILS ===")
for nm in ("rm_lin", "rm_ln", "rm_mlp", "rm_set", "rm_insulin_atoms",
           "r96_empty_world", "empty_set_vector", "reconstruction_mlp",
           "repair_feature_sets", "make_insulin_graft"):
    fnx = globals().get(nm)
    if fnx is None:
        continue
    print("\nHELPER", nm)
    try:
        src = inspect.getsource(fnx)
        print(src[:1800])
    except Exception:
        print("GLOBAL NAMES:", fnx.__code__.co_names)
        print("CONSTANTS:", str(fnx.__code__.co_consts)[:650])

print("\n=== CORE TENSOR LAYOUT ===")
for nm in NAMES:
    if nm.startswith(("core.prelude", "core.blocks", "core.coda",
                      "core.c_out", "core.b_in", "core.age")):
        print(nm, SHAPES[nm])

print("\n=== SAVED FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== SMALL RECONSTRUCTION OBJECTS ===")
for nm, ob in list(globals().items()):
    if not any(t in nm.lower() for t in ("r96", "r101", "r108", "r118", "recon", "rm_", "spec")):
        continue
    if isinstance(ob, dict):
        print(nm, "dict keys", str(list(ob.keys())[:12])[:230])
    elif isinstance(ob, (np.ndarray, torch.Tensor)):
        print(nm, tuple(ob.shape))
    elif isinstance(ob, (str, int, float, bool, tuple, list)):
        print(nm, str(ob)[:230])
```