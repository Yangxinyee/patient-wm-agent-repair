The pilot slightly damaged factual fit without correcting the rollout effects, so the original weights are restored. I will recover the architecture and earlier record-based evidence before choosing a different pathway. A useful correction must preserve factual fit and carry the right dose response through rollout.

```python
from pathlib import Path
import json, numpy as np, torch
torch.set_grad_enabled(False)
torch.set_num_threads(8)

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA SCHEMA")
print(Path("/data/README.md").read_text())

notes143 = Path("/work/notes.md").read_text()
print("\nEARLIER NOTEBOOK")
print(notes143[:18000])
print("\nFORWARD SUMMARIES")
for k143 in (1, 2, 4, 5, 8, 9):
    p143 = Path(f"/work/fwd_{k143}/summary.json")
    if p143.exists():
        print(k143, p143.read_text()[:2200])

with open("/work/notes.md", "a") as f143:
    f143.write("\nPython run 143; forwards 9. Original weights remain restored. Recovering architecture, data schema, and earlier record-only causal evidence after failed FFN pilot; next correction must validate dose transport through actual rollout, not only local readout sensitivity.\n")
```