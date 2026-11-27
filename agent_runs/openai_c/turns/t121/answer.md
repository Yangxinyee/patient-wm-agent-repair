Numeric amplification and the earlier slot shear both produced only small rollout gains. I’ll check the remaining head tests and saved state estimators before choosing a different location. In particular, I want to distinguish a weak drug signal from an edit being suppressed by downstream normalization.

```python
from pathlib import Path
import inspect
import numpy as np

print(Path("/work/notes.md").read_text()[18000:24300])

print("\nSTATE ESTIMATOR CONTENTS")
for name in ["dose_directions", "head_fits", "direction_diagnostics",
             "repair_candidates", "dose_state_coeff", "dose_gain4_72",
             "gamma", "beta", "diag_gamma_new", "diag_beta_new"]:
    if name not in globals():
        continue
    v = globals()[name]
    if isinstance(v, dict):
        print(name, {
            k: list(x)[:8] if isinstance(x, dict)
            else getattr(x, "shape", type(x).__name__)
            for k, x in list(v.items())[:8]
        })
    elif isinstance(v, np.ndarray):
        print(name, v.shape, "norm", np.linalg.norm(v))

print("\nPAIRED-STATE RELATED VARIABLES")
for name, v in sorted(list(globals().items())):
    if any(s in name.lower() for s in ["state", "panel", "control", "sensor", "replay"]):
        if isinstance(v, np.ndarray):
            print(name, v.shape)
        elif isinstance(v, dict):
            print(name, list(v)[:6])
        elif callable(v):
            print(name, inspect.signature(v))
        elif hasattr(v, "shape"):
            print(name, v.shape)
```
