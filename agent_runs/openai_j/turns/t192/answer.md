The MAP correction barely improves the dose response before using up the factual-loss margin. The validated insulin edit remains our fallback. I’ll recover the architecture and causal evidence to seek a stronger MAP pathway; it must improve dose sensitivity while preserving factual fit.

```python
from pathlib import Path
import inspect, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD COLUMNS ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("=== EARLY CAUSAL EVIDENCE ===")
print(_notes[:18000])

print("=== PERSISTENT INVENTORY ===")
for _k in sorted(list(globals())):
    if _k.startswith("_"):
        continue
    _v = globals()[_k]
    if isinstance(_v, (np.ndarray, pd.DataFrame, pd.Series)):
        print(_k, type(_v).__name__, _v.shape)
    elif callable(_v) and getattr(_v, "__module__", "") == "__main__":
        try:
            print(_k, "FUNCTION", inspect.signature(_v))
        except Exception:
            print(_k, "FUNCTION")
    elif isinstance(_v, dict):
        print(_k, "DICT", len(_v), list(_v)[:12])
    elif isinstance(_v, (list, tuple)) and len(_v) < 20:
        print(_k, type(_v).__name__, repr(_v)[:220])

print("=== ACTIVE EDITS ===", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart, run 173\n"
             "- Last MAP-head frontier has weak causal improvement within the factual-loss budget.\n"
             "- Existing insulin repair remains the validated fallback; 17 forward passes used.\n"
             "- Recovering architecture, causal evidence and persistent candidate definitions before choosing a stronger MAP pathway.\n")
```
