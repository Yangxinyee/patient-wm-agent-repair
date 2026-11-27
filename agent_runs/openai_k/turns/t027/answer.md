The inventory stopped on an imported NumPy callable; it did not test reconstruction or change weights. I’ll restrict inspection to concrete data objects and check the observation encoder’s conventions. If lab glucose’s categorical code explains the mismatch, correcting it should make lab-containing bundles match as accurately as the others.

```python
import numpy as np
import pandas as pd
import torch
import inspect
import dis
from pathlib import Path

with Path("/work/notes.md").open("a") as r22_file:
    r22_file.write(
        "\n## Runs 21–22: inventory recovery\n"
        "- Run 21 stopped while inspecting an imported NumPy callable; "
        "no reconstruction test or weight edit occurred.\n"
        "- Run 22 restricts inspection to concrete data types and audits "
        "the observation encoding conventions.\n"
        "- Python runs: 22; forward passes: 4; no weights edited.\n"
    )

r22_items = list(globals().items())
print("=== ARRAYS AND TABLES ===")
for r22_name, r22_obj in r22_items:
    if r22_name.startswith(("r20_", "r21_", "r22_", "_")):
        continue
    if isinstance(r22_obj, (np.ndarray, pd.DataFrame, pd.Series, torch.Tensor)):
        print(r22_name, type(r22_obj).__name__, tuple(r22_obj.shape))

print("\n=== DICTIONARIES AND SEQUENCES ===")
for r22_name, r22_obj in r22_items:
    if r22_name.startswith(("r20_", "r21_", "r22_", "_")):
        continue
    if isinstance(r22_obj, dict):
        print(r22_name, "dict", len(r22_obj), "keys", list(r22_obj)[:5])
    elif isinstance(r22_obj, (list, tuple)):
        print(r22_name, type(r22_obj).__name__, len(r22_obj))

print("\n=== ENCODING CONVENTIONS ===")
for r22_name in [
    "world_kq_audit", "world_bin_side_audit", "action_empty_mode_audit",
    "event_kind_ids_audit", "unit_ids_audit", "rate_unit_ids_audit",
    "code_id0", "num_spec0"
]:
    if r22_name in globals():
        print(r22_name, repr(globals()[r22_name])[:1100])

print("\n=== OBSERVATION HELPER ===")
try:
    print(inspect.getsource(embed_consistent_audit))
except (OSError, TypeError):
    dis.dis(embed_consistent_audit)

print("\n=== CATEGORICAL TENSORS ===")
for r22_name in NAMES:
    if "categorical" in r22_name:
        print(r22_name, SHAPES[r22_name])

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
```