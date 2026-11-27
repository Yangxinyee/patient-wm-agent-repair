The cache contains matched factual and dose-edited head inputs, so we can test targeted corrections without changing weights yet. Increasing the existing MAP correction nearly exhausted the factual-loss budget but still underpredicted norepinephrine doubling. I’ll recover the architecture and earlier evidence to check whether a more selective correction is possible; it must improve held-out dose sensitivity without worsening factual fit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes231 = Path("/work/notes.md").read_text()
print("\nNOTEBOOK HEADINGS")
print("\n".join(line for line in notes231.splitlines() if line.startswith("#")))
print("\nEARLY EVIDENCE")
print(notes231[:12000])

print("\nRELEVANT KERNEL OBJECTS")
for name in sorted(list(globals())):
    if name.startswith(("wm_", "head177", "head224", "map192", "map224",
                        "map218", "moment218", "response", "scalar_eval",
                        "center208", "scale208", "v208", "cent208")):
        obj = globals()[name]
        if callable(obj):
            print(name, "function", getattr(obj, "__code__", None).co_varnames
                  if hasattr(obj, "__code__") else "")
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj.keys())[:15])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        else:
            print(name, type(obj).__name__, str(obj)[:150])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 231: restart audit\n"
            "- Re-reading architecture and causal evidence; no weights changed.\n"
            "- Gain-8 remains the GPU-validated fallback. Investigating selective "
            "MAP corrections rather than further uniform gain increases.\n")
```