No results are available yet. I’ll first read the architecture and record schema to see how doses enter the model and what the records can establish about their effects. I’ll use no medical assumptions. The key test will be whether the records contain dose variation that can separate drug effects from changes in patient condition.

```python
from pathlib import Path
import json

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== VOCAB ===")
print(json.dumps(VOCAB, indent=2))

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

Path("/work/notes.md").write_text(
    "# Weight-repair notebook\n\n"
    "## Objective and constraints\n"
    "- Infer this simulated ICU's drug effects from the provided records only.\n"
    "- Repair the four dose-rollout readouts while increasing factual value NLL "
    "by no more than 0.005 nats per reading.\n"
    "- No model gradients or weight optimiser; statistical fits and computed "
    "weight edits only.\n"
    "- Limits: 300 Python runs and 50 forward requests.\n\n"
    "## Run 1\n"
    "Requested architecture, data schema, vocabulary, and tensor shapes. "
    "No causal conclusions or weight edits yet.\n"
)
```