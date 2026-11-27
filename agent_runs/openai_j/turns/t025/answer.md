The architecture confirms that action edits affect the shared recurrent state, while value-head edits could leave other predictions unchanged. The output was truncated before the helpers appeared. I’ll inspect those next; checking their state layout and head inputs should expose any reconstruction mismatch.

```python
import inspect
import numpy as np
import torch

print("=== FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(?)"
        print(name, sig)

print("\n=== NON-FUNCTION VARIABLES ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or inspect.isfunction(obj) or inspect.ismodule(obj):
        continue
    if name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj)[:6])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:70])

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("attr_heads.value", "last_", "type_emb", "query_norm",
                        "core.gap", "core.flow", "core.out_norm",
                        "core.z_prior", "core.z_head", "embed_transition.value")):
        print(name, SHAPES[name])

print("\n=== STATE ARCHIVES ===")
for path in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    with np.load(path) as archive:
        print(str(path), {k: archive[k].shape for k in archive.files})

print("\n=== VOCAB TOP-LEVEL KEYS ===")
print(list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
```