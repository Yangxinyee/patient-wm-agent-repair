```python
from pathlib import Path
import re, json, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== NOTEBOOK: EARLIER EVIDENCE ===")
notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?=^## )", notes, flags=re.M)
for s in sections:
    if any(k in s.lower() for k in [
        "identif", "causal", "true effect", "target", "instrument",
        "dynamic", "dose", "kernel", "run 17", "forward 20"
    ]) and not any(f"Run {i}:" in s for i in range(190, 198)):
        print(s[:3500])
print("\n=== FORWARD 20 FILES ===")
for p in sorted(Path("/work/fwd_20").iterdir()):
    print(p.name)
    if p.suffix == ".json":
        print(p.read_text()[:6000])

print("\n=== PERSISTENT VARIABLES ===")
for k, v in list(globals().items()):
    if re.search(r"(head177|map192|kernel|target|causal|install|cal19|base|ids|candidate|backup)", k, re.I):
        if isinstance(v, np.ndarray):
            print(k, "array", v.shape, v.dtype)
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns))
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v.keys())[:25])
        elif callable(v):
            print(k, "callable")
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "length", len(v), "first", str(v[:8])[:600])
        elif isinstance(v, (int, float, str)):
            print(k, str(v)[:600])
print("\nCurrent edits:", edited())
```
