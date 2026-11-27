The records support a negative insulin effect: threshold-based estimates and history-adjusted estimates broadly agree on the next-hour response. They do not yet establish the cumulative rollout target. I will inspect the retained longitudinal kernels and edit construction; reliable lag estimates in raw-dose units would justify cumulative calibration.

```python
import re
import numpy as np
from pathlib import Path

r241_text = Path("/work/notes.md").read_text()
r241_sections = re.split(r"(?=^## )", r241_text, flags=re.M)

print("=== SELECTED ESTIMATION / CONSTRUCTION NOTES ===")
for run in (6, 88, 136, 137, 179, 185, 194, 195, 198, 200, 209):
    matches = [
        s for s in r241_sections
        if re.match(rf"## Run {run}(?:\D|$)", s)
    ]
    for s in matches[:1]:
        print(s[:650].rstrip())

print("\n=== RETAINED KERNEL VALUES ===")
for name in ("kernels137", "kernel55", "provisional_kernels", "kernel_models88"):
    obj = globals().get(name)
    print(name, type(obj).__name__)
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, dict):
                print(" ", key, "keys", list(val))
                for k, v in val.items():
                    if np.isscalar(v):
                        print("   ", k, v)
                    elif isinstance(v, np.ndarray) and v.size <= 20:
                        print("   ", k, np.round(v, 5))
            elif isinstance(val, np.ndarray) and val.size <= 20:
                print(" ", key, np.round(val, 5))
            else:
                print(" ", key, type(val).__name__, getattr(val, "shape", None))
    elif isinstance(obj, np.ndarray) and obj.size <= 30:
        print(np.round(obj, 5))

print("\n=== KEY FUNCTION DEPENDENCIES ===")
for name in ("fit_lp136", "deconvolve_lp136", "gated_innovation94",
             "gated_ne185", "factual_eval194", "dose_metrics192"):
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        print(name, "dependencies:", fn.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 241\n"
            "Recovering longitudinal identification and channel construction in bounded "
            "output. Threshold and adjusted next-hour insulin estimates support a negative "
            "effect, but cumulative targets require lag-unit verification. No weights changed.\n")
```