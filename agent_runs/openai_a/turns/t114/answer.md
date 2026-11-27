The function list is recovered, but the object scan failed on a callable with a `shape` attribute. No weights changed. I’ll restrict the scan to actual arrays and tables, then inspect the successful posterior-reconstruction helpers. The next substantive test remains agreement with saved next-state values.

```python
import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path

r100_items = list(globals().items())
r100_inventory = []
for r100_name, r100_obj in sorted(r100_items):
    if r100_name.startswith("__") or r100_name.startswith(("r97_", "r98_", "r99_", "r100_")):
        continue
    if isinstance(r100_obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        r100_inventory.append(f"{r100_name}:{tuple(r100_obj.shape)}")
    elif isinstance(r100_obj, dict):
        r100_inventory.append(f"{r100_name}:dict[{len(r100_obj)}]")
    elif isinstance(r100_obj, (list, tuple)):
        r100_inventory.append(f"{r100_name}:{type(r100_obj).__name__}[{len(r100_obj)}]")

r100_inventory_text = " | ".join(r100_inventory)
print("=== OBJECT INVENTORY ===")
print(r100_inventory_text[:3400])
if len(r100_inventory_text) > 3400:
    print("\n=== INVENTORY TAIL ===")
    print(r100_inventory_text[-2300:])
print("\nTotal inventory characters:", len(r100_inventory_text))

print("\n=== RECENT RECONSTRUCTION OBJECTS ===")
for r100_name, r100_obj in sorted(r100_items):
    if r100_name.startswith(("r96_", "_r96_", "rm_")):
        if isinstance(r100_obj, dict):
            print(r100_name, "keys:", list(r100_obj)[:18])
        elif isinstance(r100_obj, (str, int, float, bool)):
            print(r100_name, repr(r100_obj)[:120])

r100_sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
r100_selected = [
    s for s in r100_sections
    if re.search(r"(?i)\bRD\b|regression discontinu|threshold|persistence", s)
    and not re.search(r"(?i)^##.*recovery", s)
]
print("\n=== RECORD-DERIVED CAUSAL EVIDENCE ===")
print("\n".join(r100_selected)[:2400])

with Path("/work/notes.md").open("a") as r100_f:
    r100_f.write(
        "\n## Safe inventory; Python 100, forward 11/50\n"
        "- Prior inventory failed on a callable shape attribute; this was not a model result.\n"
        "- Restricting inspection to actual arrays/tables and successful reconstruction helpers.\n"
        "- No weight changes.\n"
    )
```
