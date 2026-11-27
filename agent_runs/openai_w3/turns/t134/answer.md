```python
import os, inspect, numpy as np, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== EARLY NOTEBOOK ===")
print(open("/work/notes.md").read()[:10000])
print("=== RETAINED FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, sig)
print("=== RECENT VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if k.startswith(tuple("r%d" % i for i in range(88, 122))):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:20])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "len", len(v))
        elif isinstance(v, (str, int, float, bool)):
            print(k, repr(v)[:150])
print("=== EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 122; forwards 7. Context/view replay is close, but readout/register reconstruction remains unvalidated. Recovering architecture and retained helper inventory before calibration. No weights edited.\n")
```