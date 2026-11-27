The probe increased factual NLL by only 0.00012 nats, well within the limit. It has not yet demonstrated better dose responses. I’ll recover the edit details and test-stay IDs, then check whether insulin doubling produces the larger negative response supported by the records.

```python
from pathlib import Path
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nFIRST 100 TUNING IDS")
print([int(x) for x in tuning_ids[:100]])

print("\nRELEVANT RETAINED VARIABLES")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in
           ["probe", "decoder", "resid", "head", "feature", "state", "edit"]):
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v)[:25])
        elif callable(v):
            print(k, "callable")
        elif isinstance(v, (str, int, float, list, tuple)):
            print(k, repr(v)[:350])

print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:14000])
```
