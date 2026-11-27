The pooled increase is 0.00432 nats, but its stay-bootstrap interval reaches 0.00588. That is too close to the limit for confident unseen-stay safety. The archive comparison failed because `load` returns Torch tensors; no weights changed. I’ll fix that diagnostic and inspect safer variants, looking for a small response sacrifice that buys a meaningful factual margin.

```python
import numpy as np
import torch
from pathlib import Path

def array281(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== SAVED VARIANTS ===")
for fn281 in ("validated_safe_backup268.npz", "calibrated_candidate268.npz",
              "pre_action_extrap_backup277.npz", "independent_candidate243.npz"):
    with np.load("/work/" + fn281) as z281:
        print(fn281, z281.files)
        print("max differences:",
              {k: float(np.max(np.abs(array281(load(k)) - z281[k])))
               for k in z281.files if k in SHAPES})

print("=== FRONTIERS ===")
for fn281 in ("calibration_strength_frontier265.csv", "independent_strength_frontier242.csv"):
    tab281 = pd.read_csv("/work/" + fn281)
    print(fn281, "shape", tab281.shape, "columns", tab281.columns.tolist())
    print(tab281.head(5).to_string(index=False))

print("=== RELEVANT FUNCTIONS ===")
for k281, v281 in list(globals().items()):
    if callable(v281) and hasattr(v281, "__code__") and any(
        s in k281.lower() for s in ("candidate", "calib", "strength", "independent", "soft", "head", "factual", "nll")
    ):
        c281 = v281.__code__
        print(k281, "arguments", c281.co_varnames[:c281.co_argcount],
              "references", c281.co_names[:18])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 280–281: calibrated factual safety audit\n"
            "- Current four-tensor pooled tuning dNLL +0.00431938 over 18,654 readings / 400 stays.\n"
            "- Stay-bootstrap SE 0.00081155; 95% interval [0.00270786, 0.00588005].\n"
            "- Thus observed pooled fit passes, but unseen-stay safety margin is weak.\n"
            "- Run 280 archive diagnostic failed on NumPy/Torch mixing, after loss statistics; no weights changed.\n"
            "- Recovering safer candidates and algebraic builders. Counts: 281 Python runs, 38 forward passes.\n")
```