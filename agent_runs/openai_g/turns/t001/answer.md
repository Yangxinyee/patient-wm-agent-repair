There are no results yet. I’ll first read the architecture and record schema, without changing weights. The key question is whether the records contain dose variation that separates drug effects from treatment decisions; timing and model structure will guide that test.

```python
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

import json
import numpy as np
import pandas as pd

print("\n=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

print("\n=== VOCABULARY ===")
print(json.dumps(VOCAB, indent=2))

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== TRAINING RECORD SAMPLE ===")
print(pd.read_csv("/data/train.csv", nrows=15).to_string(index=False))

with open("/work/notes.md", "w") as f:
    f.write(
        "# Weight-repair notebook\n\n"
        "## Initial plan\n"
        "No weights changed and no causal effects assumed. Read architecture and "
        "record timing first; estimate effects from training records, including "
        "confounding and identification checks. Compute only non-gradient weight "
        "edits, then test all four dose rollouts and factual NLL. "
        "Factual NLL must remain within 0.005 nats of the original.\n"
    )
```
