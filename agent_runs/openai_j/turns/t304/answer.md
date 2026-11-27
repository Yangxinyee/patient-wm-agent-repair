The calibration screen found more factual-loss headroom, but no rollout improvement has been validated. I’ll recover the architecture and candidate builders before changing weights. Confirmation requires better paired dose responses without exceeding the factual-loss limit on independent stays.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== AVAILABLE HELPERS ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and not k.startswith("_") and any(
        s in k.lower() for s in
        ("head", "hidden", "cal", "strength", "candidate", "install",
         "loss", "lik", "moment", "build", "tensor", "kernel")
    ):
        print(k, [(p.name, p.kind.name)
                  for p in inspect.signature(v).parameters.values()])

print("\n=== RELEVANT CACHED VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or not any(s in k.lower() for s in
        ("soft", "cal", "strength", "frontier", "beta", "hidden", "current",
         "original", "candidate", "target", "tuning", "holdout")):
        continue
    if isinstance(v, np.ndarray):
        print(k, "array", v.shape, v.dtype)
    elif isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape, list(v.columns)[:14])
    elif isinstance(v, dict):
        print(k, "dict", len(v), list(v.keys())[:12])
    elif isinstance(v, (str, int, float, tuple)):
        print(k, repr(v)[:180])

print("\n=== SAVED CANDIDATE STRUCTURE ===")
for fn in ("safer_candidate247.npz", "soft_hidden_betas264.npz",
           "best_combined_v1.npz"):
    with np.load("/work/" + fn) as z:
        print(fn, [(k, z[k].shape) for k in z.files][:25])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 266: restart recovery\n"
            "- Run 265 found locally feasible calibration/strength combinations; none was installed.\n"
            "- Recovering architecture, candidate builders, and cached features before GPU validation.\n"
            "- Current safer four-tensor edit remains unchanged; 32 forward passes used.\n")
print("\nCurrent edits:", edited())
```