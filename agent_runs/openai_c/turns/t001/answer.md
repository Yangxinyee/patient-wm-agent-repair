There are no results yet. I’ll read the architecture and record definitions before changing weights. The first question is whether the records contain dose variation that can distinguish drug effects from treatment decisions; simple dose–outcome correlations would not establish that.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== VOCABULARY ===")
print(json.dumps(VOCAB, indent=2))

print("\n=== TENSOR SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== TRAIN SAMPLE ===")
print(pd.read_csv("/data/train.csv", nrows=16).to_string(index=False))

print("\n=== CURRENT OPERATIONS ===")
print(ops())

PYTHON_RUNS = 1
FORWARD_PASSES = 0
Path("/work/notes.md").write_text(
    "# World-model repair notebook\n\n"
    "## Initial inspection\n"
    "- Python runs: 1; forward passes: 0.\n"
    "- No weight edits made.\n"
    "- Read architecture, record definitions, vocabulary and tensor shapes first.\n"
    "- Drug effects are unknown. No medical assumptions will be used.\n"
    "- Plan: establish effects from treatment variation in the records, inspect "
    "the model's corresponding predictions, then test targeted weight edits "
    "against both dose rollouts and the factual NLL constraint.\n"
)
```
