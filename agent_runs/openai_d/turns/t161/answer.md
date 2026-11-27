The architecture confirms that these edits affect only the value head, so we can compare their factual losses without recomputing the teacher-forced states. The recovery output was truncated before the useful comparisons. I’ll retrieve those summaries and the cached candidate variables next; this should reveal whether reducing insulin strength can recover enough factual-loss budget.

```python
import re
from pathlib import Path

notes_text = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes_text)

wanted = [
    ("## Run 3\n", 2200),
    ("## Run 11\n", 900),
    ("## Forward 10", 1000),
    ("## Runs 108", 1400),
    ("## Restart / Runs 113", 1500),
    ("## Run 116", 700),
    ("## Forward 14", 900),
]
for prefix, limit in wanted:
    matches = [s for s in sections if s.startswith(prefix)]
    print("\n" + (matches[0][:limit] if matches else "NOT FOUND: " + prefix))

print("\n=== RELEVANT CACHED VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if k.startswith(("r115", "r139", "r138", "r137")):
        if isinstance(v, dict):
            print(k, "dict keys:", list(v.keys())[:25])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        else:
            print(k, type(v).__name__, str(v)[:180])

print("\n=== USER-DEFINED FUNCTIONS ===")
print(", ".join(
    k for k, v in sorted(list(globals().items()))
    if callable(v) and getattr(v, "__module__", None) == "__main__"
))
```