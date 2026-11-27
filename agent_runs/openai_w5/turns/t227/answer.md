The saved weight edits survived the reset. The architecture confirms that dose changes pass through the action encoder and recurrent state before reaching the value head. I’ll recover the record-derived effects next, so any further calibration stays grounded in the simulated ICU rather than assumptions about the drugs.

```python
from pathlib import Path
import json, numpy as np, pandas as pd, torch
torch.set_num_threads(8)
torch.set_grad_enabled(False)

print("=== DATA README ===")
print(Path("/data/README.md").read_text()[:2200])
print("\n=== NOTEBOOK: FIRST 6500 CHARACTERS ===")
print(Path("/work/notes.md").read_text()[:6500])

V = json.loads(Path("/model/vocab.json").read_text())
print("\n=== VOCAB STRUCTURE ===")
print("Keys:", list(V))
print("Codes:", V["codes"])
print("Config:", V["config"])
print("\n=== RECOVERED FORWARD FILES ===")
print({p.name: [q.name for q in p.iterdir()]
       for p in sorted(Path("/work").glob("fwd_*"))})
```