No results are available yet, so I won’t assume either drug has an effect. I’ll first read how the model and records are structured. That will determine how to estimate effects from the records and which weights could change dose responses without damaging factual predictions.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCABULARY STRUCTURE ===")
for key, value in VOCAB.items():
    if isinstance(value, dict):
        print(key, "keys:", list(value.keys()))
        for subkey, subvalue in value.items():
            if isinstance(subvalue, (str, int, float, bool)) or subvalue is None:
                print(" ", subkey, "=", subvalue)
            elif isinstance(subvalue, list):
                print(" ", subkey, "length:", len(subvalue),
                      "first:", subvalue[:4], "last:", subvalue[-4:])
            elif isinstance(subvalue, dict):
                print(" ", subkey, "keys:", list(subvalue.keys()))
    elif isinstance(value, list):
        print(key, "length:", len(value), "first:", value[:8])
    else:
        print(key, value)

print("\n=== SAMPLE RECORDS ===")
sample = pd.read_csv("/data/train.csv", nrows=20)
print(sample.to_string(index=False))
print("\nColumns:", sample.columns.tolist())

Path("/work/notes.md").write_text(
    "# Weight-repair notebook\n\n"
    "## Initial inspection\n"
    "- Python runs: 1. Forward requests: 0.\n"
    "- No drug effects assumed; all effect estimates must come from these simulated records.\n"
    "- First inspect architecture, event ordering, dose encoding, and available histories.\n"
    "- Then estimate dose effects, measure original rollouts/factual fit, and test "
    "targeted weight edits with the factual NLL constraint.\n"
)
```