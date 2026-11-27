Uniform attention weakened the dose responses, so that edit was rejected. The sharper-attention candidate is still untested. I’ll recover the architecture, record-based effect estimates, and probe IDs before testing it; stronger responses would support this pathway only if factual fit remains within the limit.

```python
import json, pathlib, numpy as np, pandas as pd

print("ARCHITECTURE")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README")
print(pathlib.Path("/data/README.md").read_text())

notes_text = pathlib.Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK: RECORD-BASED EVIDENCE")
print(notes_text[:18000])

print("\nCURRENT OPERATIONS")
print(ops())

print("\nSAVED VARIABLES")
for key, val in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(val, np.ndarray):
        print(key, "array", val.shape)
    elif isinstance(val, pd.DataFrame):
        print(key, "DataFrame", val.shape, list(val.columns)[:12])
    elif isinstance(val, dict) and any(s in key.lower() for s in ["caus", "effect", "fit", "state", "dir", "sens", "vocab"]):
        print(key, "dict keys", list(val)[:20])
    elif isinstance(val, (list, tuple)) and any(s in key.lower() for s in ["ids", "stays"]):
        print(key, repr(val)[:2000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 95: restart recovery\n"
            "- Numeric gain 4 plus prelude query x2 remains installed and unvalidated.\n"
            "- Recovering architecture, record-based effects, saved directions and probe IDs.\n"
            "- Forward passes used: 35; 15 remain. No edits changed.\n")
```