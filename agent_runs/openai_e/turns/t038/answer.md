The END-bundle reconstruction now matches closely, but recovering raw dose is not the same as recovering causal dose variation. I’ll recover the earlier record-based estimates and architecture before selecting an edit. A credible effect should survive adjustment for how doses were assigned.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== EARLY NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:15000])

print("=== RETAINED OBJECT INVENTORY ===")
inventory_lines = []
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "notes_text", "inventory_lines", "obj"}:
        continue
    if isinstance(obj, pd.DataFrame):
        inventory_lines.append(f"{name}: DataFrame {obj.shape}; columns={list(obj.columns)[:24]}")
    elif isinstance(obj, np.ndarray):
        inventory_lines.append(f"{name}: ndarray {obj.shape} {obj.dtype}")
    elif isinstance(obj, torch.Tensor) and obj.numel() > 10:
        inventory_lines.append(f"{name}: Tensor {tuple(obj.shape)} {obj.dtype}")
    elif inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            inventory_lines.append(f"{name}: function {inspect.signature(obj)}")
        except Exception:
            inventory_lines.append(f"{name}: function")
    elif isinstance(obj, dict) and obj:
        inventory_lines.append(f"{name}: dict[{len(obj)}] keys={list(obj.keys())[:12]}")
print("\n".join(inventory_lines))

print("=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())
print("=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 35: recover causal evidence before editing\n"
            "END convention now numerically verifies drug bundles. Action encoding predicts raw dose better than condition-adjusted dose; this alone is not a causal correction. Recovering architecture and early record-based estimates. No edits. Python runs: 35; forward passes: 3.\n")
```