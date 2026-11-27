The dose alignment is correct, but the factual-error estimates are too noisy to justify an edit. I’ll recover the architecture and existing helpers to identify a dose-sensitive repair that preserves ordinary predictions. The hypothesis is that a targeted edit can improve intervention responses; rollout improvement without exceeding the factual-NLL limit would support it.

```python
from pathlib import Path
import inspect
import pandas as pd

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
for path in sorted(Path("/work").glob("fwd_*/dose_edit_original.csv")):
    frame = pd.read_csv(path)
    print(path)
    reached = frame["reached"].astype(str).str.lower().eq("true")
    print(frame.loc[reached].groupby("edit")["shift"]
          .agg(["mean", "count"]).to_string())

print("\n=== EARLY NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:10000])
```
