```python
from pathlib import Path
import numpy as np, pandas as pd, torch, time

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== TENSOR NAMES AND SHAPES ===")
for n in NAMES:
    print(n, SHAPES[n])
print("\n=== RETAINED OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape, list(v.columns)[:30])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v)[:20])
print("\n=== FORWARD FILES ===")
for p in sorted(Path("/work").glob("fwd_*/*")):
    print(p.name, str(p), p.stat().st_size)
print("Current edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 32\nDrug-posterior reconstruction remains inaccurate (RMSE approximately 0.13 versus posterior RMS 0.248). Recovering the exact architecture and retained data before choosing a repair; no weights edited.\n")
```