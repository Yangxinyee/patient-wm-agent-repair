The latest candidates look safe on local factual tests, but their norepinephrine improvements remain unverified in rollouts. I’ll recover the candidate and installation helpers before choosing a GPU test. A larger doubling effect without breaching the factual-loss limit would support this approach.

```python
import inspect, re, numpy as np, pandas as pd
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE AND EDIT HISTORY ===")
print(notes[:18000])

print("\n=== RECENT FITTING OBJECTS ===")
for name in sorted(list(globals())):
    if re.search(r"(217|218|216|212|177|192|pooled|moment|head.*make|local.*score|install)", name, re.I):
        obj = globals()[name]
        if callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                print(name, type(obj).__name__)
            try:
                print(inspect.getsource(obj)[:14000])
            except Exception:
                pass
        elif isinstance(obj, pd.DataFrame):
            print(name, "DataFrame", obj.shape, list(obj.columns))
        elif isinstance(obj, dict):
            keys = list(obj.keys())
            print(name, "dict", len(keys), "keys:", keys[:30])
            for key in keys[:2]:
                value = obj[key]
                print(" ", key, type(value).__name__,
                      getattr(value, "shape", None))
        elif isinstance(obj, (np.ndarray, list, tuple)):
            print(name, type(obj).__name__, getattr(obj, "shape", None),
                  "length", len(obj))
        else:
            print(name, type(obj).__name__, repr(obj)[:500])

print("\n=== ACTIVE EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 219: restart inspection\n"
            "- Pooled/rescaled frontier has locally factual-safe candidates; "
            "rollout improvements remain unverified.\n"
            "- No weights changed. Recovering installation and scoring helpers "
            "before a matched GPU validation.\n")
```