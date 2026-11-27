The paired states are available, but the norepinephrine embedding change was only a diagnostic probe. I’ll restore it now and inspect the state differences. A consistent response after norepinephrine events would support using those differences to design a targeted correction; negligible or unrelated changes would argue against that approach.

```python
import os, json, inspect
import numpy as np
import pandas as pd
import torch

drop_edit("embed.code.weight")
print("Remaining edits:", edited())

print("\n--- ARCHITECTURE ---")
print(open("/model/ARCHITECTURE.md").read())
print("\n--- DATA README ---")
print(open("/data/README.md").read())

r125_original = dict(np.load("/work/fwd_15/states_original.npz"))
r125_probe = dict(np.load("/work/fwd_15/states_edited.npz"))
print("\nState arrays:",
      {k: (v.shape, str(v.dtype)) for k, v in r125_original.items()})

for k in ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"):
    assert np.array_equal(r125_original[k], r125_probe[k]), k

r125_dx = r125_probe["x"] - r125_original["x"]
r125_ne = r125_original["has_norepinephrine"].astype(bool)
for label, mask in [
    ("all", np.ones(len(r125_dx), dtype=bool)),
    ("NE bundles", r125_ne),
    ("other bundles", ~r125_ne),
]:
    print(label, "rows", int(mask.sum()), "mean delta norms",
          [round(float(np.linalg.norm(r125_dx[mask, a:b], axis=1).mean()), 6)
           for a, b in [(0,192), (192,384), (384,576),
                        (576,768), (768,960), (960,992)]])

print("\n--- Existing helper functions ---")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, inspect.signature(v))
        except (TypeError, ValueError):
            print(k)

print("\n--- Recent working variables ---")
prefixes = ("r124", "r123", "r122", "r121", "r120",
            "r119", "r118", "r117", "r115")
for k, v in sorted(list(globals().items())):
    if k.startswith(prefixes):
        if isinstance(v, dict):
            print(k, "dict keys:", list(v)[:35])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, (str, int, float, list, tuple)):
            print(k, repr(v)[:350])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 125: probe recovery\n"
            "Removed the diagnostic embed.code.weight edit immediately after "
            "forward 15. The four strength-2 insulin value-head edits remain. "
            "Loaded aligned original/probed states to investigate a controlled "
            "NE response direction; no NE repair has been accepted.\n")
```