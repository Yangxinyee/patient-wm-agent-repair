The trial’s factual NLL increased by 0.00471, narrowly within the 0.005 limit. Its stronger norepinephrine correction still needs rollout validation. I’ll recover the record-derived targets and checkpoint details first; a larger MAP response with preserved factual fit would support keeping this trial.

```python
from pathlib import Path
import re
import pandas as pd
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes)
target_sections = [
    s for s in sections
    if any(w in s.lower() for w in (
        "causal target", "identification", "record-derived",
        "true effect", "long-horizon", "target estimate"
    ))
]
print("\n=== RECORD-DERIVED TARGET HISTORY ===")
for s in target_sections[:8]:
    print(s[:1800])
print("\n=== RECENT CHECKPOINTS ===")
print(notes[-3200:])

print("\n=== FORWARD FILE SCHEMAS AND STAY IDS ===")
for k, kind in [(29, "dose_edit"), (32, "factual")]:
    p = Path(f"/work/fwd_{k}/{kind}_original.csv")
    df = pd.read_csv(p)
    print(str(p), df.shape, list(df.columns))
    for c in df.columns:
        if "stay" in c.lower():
            ids = df[c].drop_duplicates().tolist()
            print(c, "count:", len(ids), "first:", ids[:12], "last:", ids[-5:])
    print(df.head(2).to_string(index=False))

print("\n=== RELEVANT KERNEL OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith(("r235", "r240", "r241", "r242", "r232")):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj.keys())[:30])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif callable(obj):
            print(name, "callable parameters:", list(inspect.signature(obj).parameters))
        else:
            print(name, type(obj).__name__)
print("Currently edited:", edited())
```
