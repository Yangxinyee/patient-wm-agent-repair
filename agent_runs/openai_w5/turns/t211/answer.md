The architecture confirms that dose edits enter through the action encoder and recurrent state, so a locally calibrated value head is not enough by itself. The diagnostic output was truncated. I’ll recover the record-derived effect estimates and the latest replay objects before changing anything; this should let us distinguish transport attenuation from a head-calibration error.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== RECORD EVIDENCE: NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:5300])

print("=== RECENT KERNEL OBJECTS ===")
recent_names = list(globals().keys())[-110:]
for name in recent_names:
    obj = globals().get(name)
    if name.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:8])
    elif inspect.isfunction(obj):
        print(name, str(inspect.signature(obj)))
    elif isinstance(obj, (str, int, float, bool, type(None))):
        print(name, repr(obj)[:100])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))

print("=== FUNCTION NAMES ===")
print(" ".join(name for name, obj in list(globals().items())
               if inspect.isfunction(obj)))
print("=== MEASUREMENT ENCODER HELPER SOURCE ===")
for name in ["measurement_atom_local", "set_encode_single_local",
             "wm_step_boundary176"]:
    fn = globals().get(name)
    if fn is not None:
        try:
            print(name, inspect.getsource(fn)[:1800])
        except Exception:
            print(name, "source unavailable; globals:",
                  [x for x in fn.__code__.co_names if x in globals()])
```
