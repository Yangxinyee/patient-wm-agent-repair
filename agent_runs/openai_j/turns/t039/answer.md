```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

print("\n=== KERNEL HELPERS ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if inspect.isfunction(obj):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name, "(signature unavailable)")
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, getattr(obj, "shape", None))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:12])

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== ORIGINAL ROLLOUT SUMMARY ===")
import pandas as pd
for path in sorted(Path("/work").glob("fwd_*/dose_edit_original.csv")):
    frame = pd.read_csv(path)
    print(path)
    print(frame.loc[frame["reached"].astype(str).str.lower().eq("true")]
          .groupby("edit")["shift"].agg(["mean", "count"]).to_string())

print("\n=== EARLY NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:10000])
```
