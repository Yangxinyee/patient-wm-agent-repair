The architecture explains why edits to an intermediate state may have little effect after normalization and the reading registers. The inspection was truncated, so I’ll retrieve the causal estimates, saved candidate variables, and factual score compactly. Those results will determine whether to retain or remove the current head pilot.

```python
from pathlib import Path
import json, numpy as np, pandas as pd

print("NOTEBOOK IDENTIFICATION SECTION")
print(Path("/work/notes.md").read_text()[4500:10500])

print("\nFUNCTIONS")
print([k for k, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\nDICTIONARIES")
for k, v in list(globals().items()):
    if isinstance(v, dict) and k not in {"__builtins__", "SHAPES", "VOCAB"}:
        print(k, str(list(v.keys()))[:180])
print("\nARRAY/DATAFRAME NAMES")
print([(k, v.shape) for k, v in list(globals().items())
       if isinstance(v, (np.ndarray, pd.DataFrame))])

print("\nCURRENT OPS")
print(str(ops())[:1500])
print("\nFACTUAL PASS 16")
for p in sorted(Path("/work/fwd_16").glob("*")):
    print(p.name)
    if p.suffix == ".csv":
        d = pd.read_csv(p)
        print("shape", d.shape, "columns", d.columns.tolist())
        for c in d.columns:
            if "nll" in c.lower():
                print(c, float(d[c].mean()))
    elif p.suffix == ".json":
        j = json.loads(p.read_text())
        print(str(j)[:1800])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 32\n"
            "- Architecture recovered; run 31 output truncated before the saved variables and factual result.\n"
            "- Retrieving compact evidence and candidate inventory. Edits unchanged; forward passes: 17.\n")
```
