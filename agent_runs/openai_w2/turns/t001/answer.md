There are no results yet. I’ll first read the architecture and record schema, without assuming either drug has a real-world effect. Next I’ll look for dose variation that can identify effects from the records and a weight-editing route that preserves factual predictions. If those are absent, I’ll revise the approach rather than impose a drug response.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

def compact(x):
    if isinstance(x, dict):
        return {k: compact(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) > 20:
            return {"length": len(x), "first": x[:8], "last": x[-4:]}
        return x
    return x

print("\n=== VOCABULARY ===")
print(json.dumps(compact(VOCAB), indent=2))

print("\n=== FIRST TRAINING ROWS ===")
print(pd.read_csv("/data/train.csv", nrows=16).to_string(index=False))

session = {"python_runs": 1, "forward_passes": 0}
Path("/work/notes.md").write_text(
    "# World-model weight repair\n\n"
    "## Constraints\n"
    "- Infer drug effects solely from the supplied simulated records.\n"
    "- No model gradients or weight optimiser.\n"
    "- Preserve factual value NLL within +0.005 nats of original.\n"
    "- Maximum 300 Python runs and 50 forward requests.\n\n"
    "## Initial plan\n"
    "Read architecture and schema; examine treatment assignment and temporal "
    "variation; estimate effects with falsification checks; construct a "
    "targeted weight edit; validate factual fit and the exact rollout.\n\n"
    "## Run 1\n"
    "Inspecting architecture, tensor shapes, vocabulary, and sample records. "
    "No effect estimates or edits yet.\n"
)
print("\nCOUNTS:", session)
print("CURRENT EDITS:", edited())
```