The latest test found weight-edit directions that move glucose and MAP nearly independently. That gives us a usable mechanism, but the nonzero factual residuals could still hurt prediction quality. I’ll recover the exact construction and record-based effect estimates before testing a small, reversible edit. It succeeds only if rollout effects improve while factual NLL stays within budget.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_recover = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes_recover[:16000])

print("\n=== KERNEL VARIABLE NAMES ===")
print(" ".join(sorted(k for k in globals() if not k.startswith("_"))))

print("\n=== RECENT EDIT INGREDIENTS ===")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in
           ["120", "121", "122", "123", "124", "125", "126", "127", "128",
            "prelude", "resid", "repair", "steer", "bench", "target"]):
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", None),
              list(v.keys())[:35] if isinstance(v, dict) else "")
print("\nCurrent edits:", edited())
```