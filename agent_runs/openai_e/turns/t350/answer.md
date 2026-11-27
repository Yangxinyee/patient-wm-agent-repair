Candidate B substantially improves insulin responses, but norepinephrine effects remain much smaller than the record-derived estimates. Its observed factual NLL increase passes the constraint, with limited statistical safety margin. I’ll review the evidence and safer archived edits before choosing the final candidate. A safer edit retaining most of the insulin improvement would support modest shrinkage; a large loss of benefit would argue against it.

```python
from pathlib import Path
import numpy as np
import re

arch = Path("/model/ARCHITECTURE.md").read_text()
notes = Path("/work/notes.md").read_text()
print("=== ARCHITECTURE ===")
print(arch)
print("\n=== NOTEBOOK SIZE ===", len(notes))
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(notes[:5500])

print("\n=== ARCHIVE CONTRACTS ===")
for fn in ["joint236.npz", "joint274.npz", "joint283.npz", "joint284.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as z:
        print(fn, {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
        for k in z.files:
            if z[k].size <= 12:
                print(" ", k, z[k])

print("\n=== REUSABLE KERNEL OBJECTS ===")
print([(k, type(v).__name__, getattr(v, "shape", None))
       for k, v in list(globals().items())
       if not k.startswith("_") and
       re.search(r"target|causal|beta|delta|direction|component|coef|train|tuning|dose|basis|joint", k, re.I)][:100])
print("\n=== CURRENT EDITS ===", edited())
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 289: final-budget evidence audit\n"
        "Forward47 candidate B reached-hour effects: insulin x2=-11.312, half=5.383; "
        "NE x2=.411, half=-.720. B all400 factual cost=.0036534041, "
        "one-sided95 upper=.0050963395. Reviewing architecture, record-only "
        "identification and archived safer alternatives before allocating three "
        "remaining forwards. No weights changed.\n"
        "Python runs: 289; forwards: 47.\n"
    )
```