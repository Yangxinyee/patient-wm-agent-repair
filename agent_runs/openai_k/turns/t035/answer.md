The record-based estimates suggest persistent insulin effects and shorter-lived norepinephrine effects; the original rollouts appear too weak. That supports investigating a targeted correction, not indiscriminately amplifying doses. I’ll now isolate the recurrent reconstruction mismatch. Agreement with saved states is the necessary check before using this pathway.

```python
import inspect
import numpy as np
import torch

print("=== CORE TENSORS ===")
for rr30_name in NAMES:
    if rr30_name.startswith("core."):
        print(rr30_name, SHAPES[rr30_name])

print("\n=== RECONSTRUCTION HELPERS ===")
for rr30_name in ("audit_attention", "audit_block", "audit_core_update"):
    rr30_fn = globals().get(rr30_name)
    if rr30_fn is not None:
        print("\n", rr30_name)
        try:
            print(inspect.getsource(rr30_fn))
        except (OSError, TypeError):
            print("Source unavailable.")
            print("arguments:", rr30_fn.__code__.co_varnames[:rr30_fn.__code__.co_argcount])
            print("local names:", rr30_fn.__code__.co_varnames)
            print("constants:", rr30_fn.__code__.co_consts)

print("\n=== TARGETED OBJECTS ===")
for rr30_name, rr30_obj in list(globals().items()):
    if rr30_name.startswith(("core_probe", "state_", "bundle_", "post_", "full_post")):
        rr30_typ = type(rr30_obj).__name__
        if rr30_typ in ("ndarray", "Tensor", "DataFrame"):
            print(rr30_name, rr30_typ, tuple(rr30_obj.shape))
        elif rr30_typ in ("dict", "list", "tuple"):
            print(rr30_name, rr30_typ, len(rr30_obj))
        elif not callable(rr30_obj):
            print(rr30_name, rr30_typ, str(rr30_obj)[:160])

print("\n=== SAVED STATE FILE STRUCTURE ===")
from pathlib import Path
for rr30_path in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    with np.load(rr30_path, allow_pickle=False) as rr30_archive:
        print(str(rr30_path))
        for rr30_key in rr30_archive.files:
            rr30_arr = rr30_archive[rr30_key]
            print(" ", rr30_key, rr30_arr.shape, rr30_arr.dtype)

with open("/work/notes.md", "a") as rr30_f:
    rr30_f.write("\n## Run 30: recurrent-path audit\n"
                 "- Record-derived impulses suggest persistent insulin effects and faster MAP-effect decay.\n"
                 "- Recovering exact recurrent tensor layout and existing helpers to resolve state mismatch.\n"
                 "- Python runs: 30; forward passes: 4; no weights edited.\n")
```