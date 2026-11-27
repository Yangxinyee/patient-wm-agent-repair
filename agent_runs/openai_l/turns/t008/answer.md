The two analyses disagree on the effects’ signs, so the state-space estimates are not yet credible rollout targets. I’ll recover the architecture and analysis code, then check timing and confounding. Agreement across analyses that use different sources of dose variation would support an effect estimate; persistent disagreement would refute it.

```python
from pathlib import Path
import numpy as np, pandas as pd, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== PERSISTENT ANALYSIS VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape, "columns:", list(v.columns))
    elif isinstance(v, np.ndarray):
        print(k, "ndarray", v.shape, str(v.dtype))
    elif isinstance(v, dict) and any(s in k.lower() for s in ["fit", "ssm", "target", "panel", "state"]):
        print(k, "dict keys:", list(v)[:40])
    elif callable(v) and any(s in k.lower() for s in ["fit", "ssm", "panel", "kalman", "loss", "feature"]):
        print("FUNCTION", k)
        try:
            print(inspect.getsource(v))
        except (OSError, TypeError):
            print("source unavailable; signature:", inspect.signature(v))
print("\n=== VOCAB ===")
print(VOCAB)
print("\n=== EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / run 6\nSSM and rich-history coefficients disagree in sign. Treat SSM rollout targets as unvalidated, not causal truth. Recovering architecture, timing, and persistent analyses before choosing an edit.\n")
```