The decoder fits have substantial factual activation and unreliable responses to unseen dose changes. They are not yet a safe repair. I’ll recover the architecture, causal estimates, and cached helpers before editing; the next candidate must track dose changes while remaining nearly inactive on factual records.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("ACTIVE EDITS:", edited())
print("\nARCHITECTURE:\n", Path("/model/ARCHITECTURE.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("\nEARLIER NOTEBOOK:\n", notes_text[:22000])
print("\nKERNEL VARIABLES:")
for name in sorted(list(globals())):
    if name.startswith("_") or name in {"In", "Out", "get_ipython"}:
        continue
    obj = globals()[name]
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", str(keys[:18])[:500])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(name, "function")
```
