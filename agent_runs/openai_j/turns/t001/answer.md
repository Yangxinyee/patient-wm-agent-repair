There are no results yet. I’ll inspect the architecture and record format first, then estimate drug effects from the records without assuming either drug has a medical effect. This will show what evidence is available and which weights could be edited selectively.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSOR INVENTORY ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
for key, value in VOCAB.items():
    if isinstance(value, dict):
        print(key, "dict keys:", list(value.keys()))
    elif isinstance(value, list):
        print(key, "list length:", len(value), "first entries:", value[:8])
    else:
        print(key, value)

print("\n=== TRAIN SAMPLE ===")
sample = pd.read_csv("/data/train.csv", nrows=24)
print(sample.dtypes)
print(sample.to_string(index=False))

print("\n=== TUNING SAMPLE ===")
print(pd.read_csv("/data/tuning.csv", nrows=6).to_string(index=False))

Path("/work/notes.md").write_text(
    "# Weight-repair notebook\n"
    "## Initial plan\n"
    "- Inspect architecture and data before choosing an edit.\n"
    "- Infer drug effects solely from the supplied simulated-ICU records.\n"
    "- Distinguish treatment-selection associations from treatment effects.\n"
    "- Use statistical fits and algebraic weight edits only; no gradients or optimiser.\n"
    "- Validate factual value NLL and rollout effects, including cross-sensor consistency.\n"
    "- No findings or edits yet.\n"
)
print("\nCurrent edits:", edited())
```