The reconstruction closely matches the model’s output, and direct recurrent-input perturbations have much more influence than the posterior-latent perturbations did. That makes this pathway worth testing, but it does not yet establish a safe edit. I’ll recover the fitted dose residuals and implementation details; the key test will be whether they change dose responses while barely affecting factual predictions.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
notes121 = Path("/work/notes.md").read_text()
print(notes121[:18000])

print("\n=== KERNEL FUNCTIONS ===")
print(sorted(k for k, v in list(globals().items())
             if callable(v) and not k.startswith("_")))

print("\n=== RELEVANT KERNEL VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or callable(v):
        continue
    if any(s in k.lower() for s in (
        "120", "119", "118", "117", "116", "115", "114", "113", "112",
        "prelude", "resid", "probe", "repair", "target", "effect", "slope",
        "b_in", "affine", "latent", "fit", "jac", "direction"
    )):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, v.shape)
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v.keys())[:35])
        elif isinstance(v, (str, int, float, bool, type(None))):
            print(k, repr(v)[:500])
        else:
            print(k, type(v).__name__, repr(v)[:250])
```
