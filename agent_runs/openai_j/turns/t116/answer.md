The last result confirms that the rejected edit changed GPU predictions far more than our local calculation expected. No edits remain. I’ll recover the architecture and causal-estimation results before trying another repair. The next candidate must first reproduce the original GPU predictions row by row; otherwise its apparent factual safety is not trustworthy.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE (read first) ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK: IDENTIFICATION AND FIRST PROBES ===")
print(notes_text[:26000])

for filename in ["dml_results.csv", "dynamic_iv.csv", "sensor_iv.csv",
                 "proxy_iv.csv", "local_projections.csv", "kernel_targets.csv"]:
    path = Path("/work") / filename
    if path.exists():
        table = pd.read_csv(path)
        print(f"\n=== {filename}: shape={table.shape} ===")
        print(table.head(35).to_string(index=False))

print("\n=== ORIGINAL FACTUAL OUTPUT FORMAT ===")
print(pd.read_csv("/work/fwd_4/factual_original.csv").head(8).to_string(index=False))
print("\n=== ACTIVE EDITS ===")
print(edited())
```
