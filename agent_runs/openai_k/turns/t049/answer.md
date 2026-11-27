The prior-mean reconstruction is only approximate, so it is not yet reliable for calibrating a weight edit. The records suggest different response durations for glucose and MAP. Next I’ll recover the update equations and existing helpers; a usable correction must reproduce those dynamics without materially changing factual predictions.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14500])
print("\n=== RETAINED HELPERS AND ARRAYS ===")
for k in sorted(list(globals())):
    if k.startswith(("rr", "r3", "r4")):
        v = globals()[k]
        if callable(v):
            import inspect
            try:
                desc = str(inspect.signature(v))
            except Exception:
                desc = "callable"
        elif hasattr(v, "shape"):
            desc = f"{type(v).__name__}, shape={v.shape}"
        elif isinstance(v, dict):
            desc = f"dict, keys={list(v)[:15]}"
        elif isinstance(v, (list, tuple)):
            desc = f"{type(v).__name__}, len={len(v)}"
        else:
            desc = type(v).__name__
        print(k, desc)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 44: recover edit implementation context\n"
            "- Prior-mean head reconstruction is approximate, not an exact forward surrogate.\n"
            "- Recovering architecture, identification evidence, and retained helpers before editing.\n"
            "- Python runs: 44; forward passes: 4; no tensors edited.\n")
```