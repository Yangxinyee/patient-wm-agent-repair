The records support substantial insulin lowering and positive norepinephrine effects, unlike the original rollout. The output was truncated before the helper inventory, so I’ll retrieve that compactly. This should identify the replay mismatch before any weights are changed.

```python
import inspect
import numpy as np
import torch

print("=== CALLABLE HELPERS ===")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, str(inspect.signature(v))[:180])
        except Exception:
            print(k)

print("\n=== RECENT OBJECTS ===")
for k in sorted(list(globals())):
    if not k.startswith(("r101", "r102", "r103", "r104", "r105")):
        continue
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:16])
    elif isinstance(v, str):
        print(k, "string", len(v), repr(v[:90]))
    elif callable(v):
        print(k, "callable")
    else:
        print(k, type(v).__name__, repr(v)[:100])

print("\n=== SAVED SOURCE STRINGS ===")
for k in sorted(list(globals())):
    v = globals()[k]
    if isinstance(v, str) and ("def " in v or "lambda " in v):
        print(k, len(v), repr(v[:100]))

print("\n=== REFERENCE OBJECT NAMES ===")
print([k for k in sorted(globals())
       if any(s in k.lower() for s in
              ("state", "factual", "register", "readout", "replay", "kernel", "policy"))][:100])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 107; forwards 7. Calibration is not yet trusted: "
            "local readout and core disagreement require exact replay diagnosis. "
            "Original weights remain unchanged; recovering compact helper inventory.\n")
print("CURRENT EDITS", edited())
```
