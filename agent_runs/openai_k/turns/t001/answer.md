I have no evidence yet about either drug’s effect. I’ll first read the architecture and record schema, then look for treatment variation that could identify effects without relying on medical assumptions. The architecture will also show where a weight edit could change dose responses without damaging factual predictions.

```python
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import scipy
import torch

torch.set_num_threads(8)
torch.set_grad_enabled(False)

arch = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch)

print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
def summarize_vocab(x, depth=0):
    if isinstance(x, dict):
        return {k: summarize_vocab(v, depth+1) for k, v in x.items()}
    if isinstance(x, list):
        return {"length": len(x), "first": x[:5], "last": x[-3:]}
    return x
print(json.dumps(summarize_vocab(VOCAB), indent=2))

sample = pd.read_csv("/data/train.csv", nrows=16)
print("\n=== RECORD SAMPLE ===")
print(sample.to_string(index=False))
print("\n=== COLUMN TYPES ===")
print(sample.dtypes.to_string())

run_count = 1
forward_count = 0
Path("/work/notes.md").write_text(
    "# Weight-repair notebook\n\n"
    "## Run 1: initial inspection\n"
    "- Read architecture, data schema, tensor shapes, vocabulary, and a record sample.\n"
    "- No drug-effect conclusion or weight edit yet.\n"
    "- All causal conclusions will come from the supplied records.\n"
    "- Python runs: 1; forward passes: 0.\n"
)
```
