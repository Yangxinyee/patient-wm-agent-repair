The corrected targets now match the rollout’s eligible reading-hours, and they remain much larger than the original model’s responses. Candidate A is still untested. I’ll recover the architecture and saved analysis before evaluating it; it succeeds only if dose responses improve while factual NLL stays within the 0.005 limit.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

print("\n=== EARLIER NOTEBOOK CONTEXT ===")
print(Path("/work/notes.md").read_text()[:9000])

print("\n=== PERSISTENT VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape, list(v.columns)[:25])
    elif isinstance(v, np.ndarray):
        print(k, "ndarray", v.shape, v.dtype)
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v)[:30])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif callable(v):
        print(k, "callable")
    elif isinstance(v, (str, int, float, np.number)):
        print(k, repr(v)[:180])

print("\n=== TENSOR SHAPES ===")
for n in NAMES:
    print(n, SHAPES[n])

print("\n=== EXISTING FORWARD FILES ===")
for d in sorted(Path("/work").glob("fwd_*")):
    print(d.name, [p.name for p in d.iterdir()])
print("\nCurrent operation count:", len(ops()))
```