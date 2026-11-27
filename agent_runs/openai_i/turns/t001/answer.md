No results are available yet, so I won’t assume either drug has an effect. I’ll first inspect the architecture and record format, then look for evidence that separates treatment effects from dose selection. An effect that disappears after accounting for dose selection would challenge a naive causal interpretation.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
torch.set_grad_enabled(False)

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

def compact(obj):
    if isinstance(obj, dict):
        return {k: compact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 16:
            return {
                "length": len(obj),
                "first": obj[:6],
                "last": obj[-6:]
            }
        return obj
    return obj

print("\n=== VOCABULARY / NORMALIZATION / BINS ===")
print(json.dumps(compact(VOCAB), indent=2))

print("\n=== TRAINING RECORD SAMPLE ===")
sample = pd.read_csv("/data/train.csv", nrows=12)
print(sample.to_string(index=False))
print("Column dtypes:", sample.dtypes.astype(str).to_dict())

work = Path("/work")
work.mkdir(exist_ok=True)
RUN_COUNTS = {"python": 1, "forward": 0}
with (work / "notes.md").open("a") as f:
    f.write(
        "\n# Weight-repair investigation\n"
        "## Initial inspection — Python 1; forward 0\n"
        "- No weights changed and no drug effects assumed.\n"
        "- Inspect architecture, vocabulary, and record schema first.\n"
        "- Establish effects from training records, accounting for dose selection.\n"
        "- Validate any edit against paired rollouts and the factual NLL limit "
        "(original + 0.005 nats per reading).\n"
    )
print("\nCurrent edits:", edited())
```
