The regression offers a possible norepinephrine correction, but its local scores do not establish rollout accuracy or factual fit. I’ll recover the candidate definitions and record-derived targets before testing one. Confirmation requires better norepinephrine readouts, preserved insulin responses, and an NLL increase no greater than 0.005.

```python
from pathlib import Path
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes145 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-EVIDENCE NOTES")
print(notes145[:8500])

print("\nLIVE OBJECTS FROM RUNS 143–144")
for name145 in sorted(list(globals())):
    if name145.endswith(("143", "144")):
        obj145 = globals()[name145]
        if hasattr(obj145, "shape"):
            print(name145, type(obj145).__name__, tuple(obj145.shape))
        elif isinstance(obj145, dict):
            print(name145, "dict", {
                str(k): (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                for k, v in list(obj145.items())[:20]
            })
        elif isinstance(obj145, (list, tuple)):
            print(name145, type(obj145).__name__, "length", len(obj145))
        elif callable(obj145):
            print(name145, "callable")
        elif isinstance(obj145, (str, int, float, bool, np.number)):
            print(name145, repr(obj145)[:300])

print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f145:
    f145.write(
        "\n## Run 145: restart recovery\n"
        "Recovered architecture, early record-evidence notes, and live regression "
        "objects before any new candidate installation. Run144 local frontier "
        "is not a rollout or factual-NLL validation. No weights changed. "
        "Python runs: 145; forwards: 15.\n"
    )
```