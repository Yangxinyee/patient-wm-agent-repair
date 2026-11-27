The live objects suggest the current head uses stronger settings than the notebook’s last validated candidate: insulin 0.5 and norepinephrine 0.4, versus 0.3 each. I’ll recover its fit statistics and recent rollout results before changing it. This will distinguish a useful improvement from extra factual distortion.

```python
import inspect, dis
from pathlib import Path
import pandas as pd

print("=== RECORD-EVIDENCE NOTES ===")
print(Path("/work/notes.md").read_text()[:4000])

print("=== CURRENT FIT ===")
print("compact240 signature:", inspect.signature(compact240))
try:
    print(inspect.getsource(compact240))
except Exception:
    dis.dis(compact240)
print("st242 architecture:", {k: v for k, v in st242["arch"].items()
                              if k in ["style", "lam", "units"]})
print("result242:", compact240(result242))
print("grid242:")
for key, val in grid242.items():
    try:
        print(key, compact240(val))
    except Exception:
        print(key, str(val)[:200])

print("=== RECENT FORWARD FILES ===")
for k in [15, 16, 17, 18]:
    folder = Path(f"/work/fwd_{k}")
    print(folder.name, [p.name for p in folder.iterdir()] if folder.exists() else "absent")
    for p in folder.glob("dose_edit*.csv"):
        table = pd.read_csv(p)
        print(p.name, "shape", table.shape, "columns", list(table.columns))
        print(table.head(2).to_string(index=False))

print("=== AUDIT STAYS ===")
print("audit_ids236 =", list(audit_ids236))
```
