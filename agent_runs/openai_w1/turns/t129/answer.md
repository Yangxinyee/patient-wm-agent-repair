The aligned context, observation, and action inputs agree numerically. That shifts suspicion from atom encoding to a stale or incorrectly assembled posterior input. I’ll recompute the posterior from these verified components and recover the earlier readout fits. Numerical agreement would let us move on to weight editing.

```python
import numpy as np, torch, re
from pathlib import Path

rr118_input = torch.cat((rr_best_ctx111, rr117_e, rr117_u), dim=1)
rr118_z = rr_mlp(rr118_input, "z_post", "gelu")[:, :32]
rr118_oldz = r79_mlp(rr118_input, "z_post", "gelu")[:, :32]
print("Fresh/cached posterior implementation RMSE:",
      float((rr118_z - rr118_oldz).square().mean().sqrt()))
rr118_err = (rr118_z - rr_zexport4).square().mean(1).sqrt().numpy()
for label, mask in (
    ("reading", np.asarray(rr_has_reading4, bool)),
    ("active drug", np.asarray(rr_has_active4, bool)),
    ("other", ~np.asarray(rr_has_reading4, bool) & ~np.asarray(rr_has_active4, bool)),
    ("valid", np.asarray(rr_valid4, bool))
):
    if mask.any():
        print(label, "n", int(mask.sum()),
              "posterior RMSE quantiles", np.quantile(rr118_err[mask], [0, .5, .9, 1]))

for name in ("rr_base_u112", "rr_test_u111", "rr_u4"):
    a = globals().get(name)
    if isinstance(a, torch.Tensor) and a.shape == rr117_u.shape:
        d = (a - rr117_u).square().mean(1).sqrt().numpy()
        print(name, "difference from verified action input",
              np.quantile(d, [0, .5, .9, 1]))

print("\n=== EARLIER READOUT FITS ===")
sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
budget = 4400
for s in sections:
    title = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"\bRun (2[6-9]|3[0-5]|4[5-9]|5[0-9]|6[0-5])\b", title):
        chunk = s[:min(650, budget)]
        print(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break

print("\n=== READOUT-FEATURE INVENTORY ===")
for n, v in sorted(list(globals().items())):
    if n.startswith(tuple(f"r{i}_" for i in range(26, 36))) or n.startswith(("rr51_", "rr58_")):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "dict", len(v), list(v)[:4])
        elif callable(v) and hasattr(v, "__code__"):
            print(n, "function", v.__code__.co_names)
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, len(v))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 118\nAligned fresh context, observation, and action inputs agree with validated "
            "arrays. Recomputed posterior from these components to distinguish stale inputs from "
            "encoder defects. No edits. Counts: 118 Python runs, 6 forward passes.\n")
```
