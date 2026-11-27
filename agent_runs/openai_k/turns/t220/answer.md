```python
from pathlib import Path
import numpy as np
import pandas as pd
import json

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== NOTEBOOK: EARLY EVIDENCE ===")
_notes200 = Path("/work/notes.md").read_text()
print(_notes200[:6500])
print("\n=== RETAINED VARIABLE NAMES ===")
print(" ".join(sorted(k for k in globals()
                      if not k.startswith("_") and k not in {"In", "Out", "get_ipython"})))
print("\n=== FORWARD 13 FILES ===")
print([p.name for p in Path("/work/fwd_13").iterdir()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 200: restart after sign-reversal validation\n"
            "- Forward 14: reversed routing modestly improved insulin "
            "(x2 -2.682, half +1.339) but NE remained wrong "
            "(x2 -0.069, half +0.369).\n"
            "- Recovering architecture and record evidence; a shared sign change "
            "is not an adequate repair. No weights changed.\n"
            "- Python runs: 200; forward passes: 14.\n")
```