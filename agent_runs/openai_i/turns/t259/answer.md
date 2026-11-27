The response directions nearly recover the estimated one-hour effects, but stronger edits introduce cross-sensor leakage. That does not yet establish rollout accuracy or factual fit. I’ll inspect the saved directions and replay setup before installation; matched factual and rollout tests will decide whether this approach works.

```python
import os, json, numpy as np
from pathlib import Path

arch = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch)

print("\n=== SAVED RESPONSE DIRECTIONS ===")
with np.load("/work/response_directions231.npz", allow_pickle=False) as f:
    for k in f.files:
        a = f[k]
        print(k, "shape", a.shape, "dtype", a.dtype,
              "norm", float(np.linalg.norm(a)) if np.issubdtype(a.dtype, np.number) else "")

print("\n=== CURRENT EDIT SHAPES ===")
for n in edited():
    print(n, SHAPES[n])

print("\n=== RETAINED RESPONSE/REPLAY OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.endswith(("228", "229", "230", "231")):
        if callable(v) and hasattr(v, "__code__"):
            print(k, "FUNCTION",
                  "args", v.__code__.co_varnames[:v.__code__.co_argcount],
                  "defaults", v.__defaults__,
                  "references", v.__code__.co_names)
        elif isinstance(v, np.ndarray):
            print(k, "ARRAY", v.shape, v.dtype)
        elif isinstance(v, dict):
            print(k, "DICT KEYS", list(v.keys())[:30])
        elif isinstance(v, (str, int, float, list, tuple)):
            s = repr(v)
            print(k, s[:350])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 232 — response calibration recovery\n"
            "- Run231 mean directions produced proximal x2 slopes near -1.95 insulin and +2.65 NE at gain 1, with nonlinear cross-sensor leakage.\n"
            "- These are replay diagnostics, not factual or rollout certification.\n"
            "- Inspecting saved directions and implementation before a controlled GPU test.\n")
```