The last check reconstructs the current edit accurately and identifies the normalized prelude features. That supports using the new training states, but does not yet show better factual fit. I’ll recover the exact fitting inputs without changing weights; matching the saved feature transformation would confirm that a refit uses the intended pathway.

```python
from pathlib import Path
import numpy as np, inspect, dis

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRELEVANT KERNEL VARIABLES")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in
           ["147", "151", "171", "173", "176", "187", "193", "194", "195",
            "196", "197", "198", "199", "prelude", "decoder", "fold"]):
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif callable(v):
            print(k, "callable", str(inspect.signature(v)))
        elif isinstance(v, (dict, list, tuple)):
            print(k, type(v).__name__, "length", len(v))
        else:
            print(k, type(v).__name__, str(v)[:250])

print("\nNOTEBOOK HEADINGS")
notes200 = Path("/work/notes.md").read_text()
for line in notes200.splitlines():
    if line.startswith("#"):
        print(line)
print("\nCURRENT EDITS", edited())
print("CURRENT EQUALS VALIDATED176",
      np.array_equal(np.asarray(load("core.b_in.weight")),
                     np.asarray(validated_B176)))
```
