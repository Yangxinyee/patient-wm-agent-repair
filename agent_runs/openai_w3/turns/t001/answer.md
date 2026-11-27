There are no results yet. I’ll first read how the model handles doses and measurements, then inspect the record format. This will identify possible weight edits without assuming either drug has an effect.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE (read first) ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== VOCAB ===")
print(json.dumps(VOCAB, indent=2))

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== TRAIN SAMPLE ===")
print(pd.read_csv("/data/train.csv", nrows=12).to_string(index=False))
print("\n=== TUNING SAMPLE ===")
print(pd.read_csv("/data/tuning.csv", nrows=5).to_string(index=False))

Path("/work").mkdir(exist_ok=True)
Path("/work/notes.md").write_text(
    "# World-model repair notebook\n"
    "Python runs: 1. Forward requests: 0.\n"
    "Goal: establish dose effects from the simulated records, then edit weights "
    "without gradients while preserving factual value NLL within +0.005 nats.\n"
    "No medical-effect assumptions. No edits made yet.\n"
    "First step: read architecture, data schema, vocabulary, and tensor shapes.\n"
)
```
