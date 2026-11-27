Most reconstructed posteriors now match closely, but errors remain concentrated around norepinephrine and a few single-event bundles. I suspect an unresolved encoding convention. I’ll inspect those exceptions and recover the transition code before attempting an edit. If correcting the convention removes the errors, that supports the hypothesis; otherwise, the reconstruction needs a broader check.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== NOTEBOOK EARLY CONTEXT ===")
print(Path("/work/notes.md").read_text()[:9500])

print("\n=== RETAINED OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, (pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, v.shape, list(v.columns)[:18] if isinstance(v, pd.DataFrame) else "")
    elif isinstance(v, np.ndarray) or torch.is_tensor(v):
        if getattr(v, "ndim", 0) > 0:
            print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        if len(v) < 60:
            print(k, "dict", list(v.keys())[:30])
    elif callable(v) and getattr(v, "__module__", None) == "__main__":
        try:
            print(k, "function", str(inspect.signature(v)))
        except Exception:
            pass
print("\n=== EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 61 (restart)\nRun-60 posterior reconstruction is exact for most multi-world and END bundles, but 100 count mismatches and concentrated NE/single-world errors remain. Recovering transition code and exception structure before committing a targeted edit. No weights edited.\n")
```
