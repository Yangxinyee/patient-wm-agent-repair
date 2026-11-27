The best layout still leaves a mismatch. The rejected head edit offers another diagnostic: its paired GPU predictions show how a known weight change actually affected the head. I will recover that edit’s coefficients and local inputs, then compare predicted and observed changes. This can distinguish bad reconstruction from ordinary sampling noise.

```python
import sys
import inspect
import numpy as np
import torch
from pathlib import Path

print("=== RECONSTRUCTION SIGNATURE ===")
print(inspect.signature(reconstruct_reading_head))

captured100 = {}
def capture100(frame, event, arg):
    if event == "return" and frame.f_code is reconstruct_reading_head.__code__:
        captured100.update(frame.f_locals.copy())

sys.setprofile(capture100)
try:
    result100 = reconstruct_reading_head(head_meta.iloc[:4], SX)
finally:
    sys.setprofile(None)

print("\n=== RECONSTRUCTION LOCAL VARIABLES ===")
for name, value in captured100.items():
    shape = getattr(value, "shape", None)
    if isinstance(shape, (tuple, torch.Size)):
        print(name, tuple(shape))
        if len(shape) == 2 and shape[1] <= 17:
            print(value[:4])
    elif isinstance(value, (str, int, float, bool)):
        print(name, value)

print("\n=== POSSIBLE STORED EDIT COEFFICIENTS ===")
shown = 0
for name, value in list(globals().items()):
    if any(s in name.lower() for s in [
        "coef", "decoder", "delta_w", "delta_b", "candidate", "repair_w",
        "repair_b", "tilt_w", "tilt_b", "raw_w", "raw_b"
    ]):
        shape = getattr(value, "shape", None)
        if isinstance(shape, (tuple, torch.Size)):
            print(name, tuple(shape),
                  "norm:", float(np.linalg.norm(np.asarray(value))))
            shown += 1
        elif isinstance(value, dict):
            print(name, "dict keys:", list(value.keys())[:6])
            shown += 1
        elif hasattr(value, "coef_"):
            print(name, type(value).__name__, np.shape(value.coef_))
            shown += 1
        elif isinstance(value, (float, int)):
            print(name, value)
            shown += 1
        if shown >= 32:
            break

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== EXAMPLE METADATA ===")
print(head_meta.head(4).to_string(index=False))
print("\n=== PAIRED REJECTED-EDIT SUMMARY ===")
print(cmp94.groupby("event")[["dnll", "dmean"]].mean().to_string())
```