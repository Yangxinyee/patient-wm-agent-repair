No inspection has run yet, so I still have no evidence about either drug’s effect. I’ll first read the architecture and record format. These should reveal where a weight edit could act and whether the records contain dose variation that can distinguish drug effects from treatment selection.

```python
from pathlib import Path
import json, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())
print("\n=== VOCABULARY ===")
print(json.dumps(VOCAB, indent=2))
print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

Path("/work").mkdir(exist_ok=True)
Path("/work/notes.md").write_text(
    "# Model repair notebook\n"
    "## Initial plan\n"
    "Infer drug effects only from simulated records. Inspect architecture first. "
    "Check treatment selection, pre-dose histories, dose variation, and overlap. "
    "Use analytic weight edits without gradients, and validate factual NLL and "
    "the exact paired dose-edit rollout.\n"
    "Python runs: 1. Forward passes: 0.\n"
)
```
