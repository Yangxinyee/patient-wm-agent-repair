The architecture has a separate action encoder, which offers a more targeted route than changing the shared reading head. I’ll recover the record-based effect estimates and original rollout results first. They should establish which responses need correction; weak identification would argue against a large edit.

```python
from pathlib import Path
import pandas as pd
import numpy as np

text = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK ===")
print(text[:5700])

print("\n=== COMPACT OBJECT NAMES ===")
snapshot = list(globals().items())
for label, predicate in [
    ("dataframes", lambda x: isinstance(x, pd.DataFrame)),
    ("dictionaries", lambda x: isinstance(x, dict)),
    ("arrays/tensors", lambda x: hasattr(x, "shape") and not isinstance(x, (pd.DataFrame, pd.Series))),
]:
    names = [n for n, x in snapshot if not n.startswith("_") and predicate(x)]
    print(label + ":", ", ".join(sorted(names)))

print("\n=== FORWARD OUTPUT FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== ORIGINAL ROLLOUT ===")
print("columns:", list(base_roll.columns))
print(base_roll.head(4).to_string(index=False))
for col in ["edit", "dose_edit", "intervention", "scenario"]:
    if col in base_roll:
        print(base_roll.groupby(col).mean(numeric_only=True).to_string())
        break

print("\n=== FACTUAL BASELINE ===")
print("columns:", list(baseline_fact.columns))
print(baseline_fact.head(2).to_string(index=False))
print("\nCurrent edits:", edited())
```