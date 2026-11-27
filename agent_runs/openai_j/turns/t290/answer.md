The latest batch worsened value NLL by only 0.00278 nats, within the 0.005 limit. That supports factual safety, but does not establish that norepinephrine doubling is accurate. I’ll recover the exact edit construction and candidate diagnostics; a further change needs both stronger intervention responses and preserved factual fit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== CURRENT EDITS ===")
print(edited())

print("\n=== RECOVERED HELPERS ===")
_g = list(globals().items())
for name, obj in _g:
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\n=== RELEVANT KERNEL OBJECTS ===")
for name, obj in _g:
    if any(s in name.lower() for s in
           ("nonlin", "hinge", "candidate", "target", "contrast",
            "install", "frontier", "calibr", "coef", "tuning_ids")):
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, obj.dtype)
        elif isinstance(obj, pd.DataFrame):
            print(name, "DataFrame", obj.shape, list(obj.columns)[:16])
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj.keys())[:20])
        elif isinstance(obj, (float, int, str, list, tuple)):
            text = repr(obj)
            print(name, text[:240])

print("\n=== NONLINEAR FRONTIER ===")
p = Path("/work/map_nonlinear_frontier.csv")
if p.exists():
    nonlinear_recovery253 = pd.read_csv(p)
    print("Shape:", nonlinear_recovery253.shape)
    print("Columns:", list(nonlinear_recovery253.columns))
    print(nonlinear_recovery253.head(8).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 253: restart and safety audit\n"
            "- Forward 31 / fwd_32 factual value dNLL +0.00278 on 100 tuning stays.\n"
            "- Current four-tensor safer edit remains installed; no weights changed.\n"
            "- Recovering architecture, exact construction, and nonlinear candidate diagnostics.\n")
```