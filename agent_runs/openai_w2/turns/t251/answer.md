The recovered notes show why raw dose associations are insufficient: insulin thresholds also changed whether glucose was recorded. The output was clipped before the candidate inventory, so I’ll recover that separately and leave the weights unchanged. Record-based target estimates and held-out validation—not the direction of association alone—must justify the edit.

```python
import inspect
import re
import numpy as np
import torch
from pathlib import Path

current_B215 = torch.as_tensor(load("core.b_in.weight")).clone()
original_B215 = torch.as_tensor(load_original("core.b_in.weight")).clone()

def summary215(name, obj):
    s = f"{name}: {type(obj).__name__}"
    if hasattr(obj, "shape"):
        s += f" {tuple(obj.shape)}"
        if isinstance(obj, np.ndarray) and obj.size <= 16:
            s += " " + np.array2string(obj, precision=5)
        elif isinstance(obj, torch.Tensor) and obj.numel() <= 16:
            s += " " + str(obj.tolist())
    elif isinstance(obj, dict):
        s += " keys=" + repr(list(obj.keys())[:18])
    elif isinstance(obj, (float, int, bool, str, np.number)):
        s += " " + str(obj)[:180]
    elif isinstance(obj, (list, tuple)):
        s += f" len={len(obj)}"
        if len(obj) <= 8:
            s += " " + repr(obj)[:200]
    return s

items215 = list(globals().items())
lines215 = ["CURRENT EDIT / SAVED CANDIDATES"]
delta215 = (current_B215 - original_B215).numpy()
lines215.append(
    f"edit norm={np.linalg.norm(delta215):.6f}; "
    f"top singular values={np.linalg.svd(delta215, compute_uv=False)[:5]}"
)
for n215 in ["validated_B176", "validated_W176", "candidate_B204",
             "candidate_W204", "fallback_B212"]:
    if n215 in globals():
        o215 = globals()[n215]
        lines215.append(summary215(n215, o215))
        if hasattr(o215, "shape") and tuple(o215.shape) == tuple(current_B215.shape):
            a215 = o215.detach().cpu().numpy() if isinstance(o215, torch.Tensor) else np.asarray(o215)
            lines215.append(f"  max difference from current={np.max(np.abs(a215-current_B215.numpy())):.7g}")

lines215.append("\nRECENT FIT OBJECTS")
for n215, o215 in items215:
    if not n215.startswith("_") and re.search(r"206|212", n215) and not callable(o215):
        lines215.append(summary215(n215, o215))

lines215.append("\nEFFECT / TARGET OBJECT NAMES")
target_names215 = [
    n for n, o in items215
    if not n.startswith("_") and not callable(o)
    and re.search(r"target|impulse|rho|causal|effect|latent", n, re.I)
]
lines215.append(", ".join(target_names215)[:2200])

lines215.append("\nHELPER SIGNATURES")
for n215, o215 in items215:
    if callable(o215) and re.search(r"record_impulse|linear_rollout|latent_record|sensor_iv|decoder|install", n215):
        try:
            lines215.append(n215 + " " + str(inspect.signature(o215)))
        except (TypeError, ValueError):
            pass

text215 = "\n".join(lines215)
print(text215[:7800])

with open("/work/notes.md", "a") as f215:
    f215.write(
        "\n## Run215: compact recovery\n"
        "Run214 output clipped before kernel inventory. Current weights retained "
        "as current_B215; original as original_B215. Compact candidate and "
        "target inventory only; no weight changes.\n"
    )
```
