The layout tests improve mean agreement, but none yet reproduces the GPU NLL. The inspection also stopped before showing the helper because a cached object’s `shape` was not a tuple. I will fix that diagnostic and inspect the inputs directly. This does not justify a new edit yet.

```python
import inspect
import dis
import io
import numpy as np
import torch

print("=== HEAD RECONSTRUCTION IMPLEMENTATION ===")
try:
    text = inspect.getsource(reconstruct_reading_head)
except (OSError, TypeError):
    stream = io.StringIO()
    dis.dis(reconstruct_reading_head, file=stream)
    text = stream.getvalue()
print(text[:6100])

print("\n=== EIGHT-COLUMN ARRAYS ===")
for name, value in list(globals().items()):
    shape = getattr(value, "shape", None)
    if isinstance(shape, (tuple, torch.Size)) and len(shape) == 2 and shape[1] == 8:
        print(name, tuple(shape))
        print(value[:2])

print("\n=== NUMERIC COLUMN NORMS ===")
w = np.asarray(load_original("last_numeric.weight"))
print(np.linalg.norm(w, axis=0))
print("column correlations:\n", np.round(np.corrcoef(w.T), 2))

print("\n=== SHORT HELPER SIGNATURES ===")
for name in [
    "wt", "ln", "activation", "mlp", "repair_changed_action",
    "repair_core_rows", "repair_tilt", "repair_centres17"
]:
    fn = globals()[name]
    print(name, inspect.signature(fn),
          "constants:", fn.__code__.co_consts[:12])

print("\n=== STATE DICTIONARY KEYS ===")
for name in ["states", "state_data", "latent_probe_state", "repair_state"]:
    if name in globals():
        value = globals()[name]
        print(name, list(value.keys()) if hasattr(value, "keys") else type(value))

print("\n=== HEAD METADATA EXAMPLES ===")
print(head_meta.head(5).to_string(index=False))
```