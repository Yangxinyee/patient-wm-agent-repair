The carrier underestimates dose changes, especially insulin, and is unreliable for stopping or tripling doses. This points to a decoding problem, not evidence of weak drug effects. I’ll inspect the retained implementation before recalibrating it; a usable carrier must recover dose contrasts across held-out stays and all tested multipliers.

```python
import numpy as np, inspect, re
from pathlib import Path

print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRECENT OBJECTS")
for name in sorted(list(globals())):
    if re.search(r"(136|145|147|154|169|171|174|175|176|177|178|179)$", name):
        obj = globals()[name]
        if isinstance(obj, dict):
            desc = [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                    for k, v in list(obj.items())[:20]]
        elif hasattr(obj, "shape"):
            desc = (tuple(obj.shape), str(getattr(obj, "dtype", "")))
        elif callable(obj):
            try:
                desc = inspect.signature(obj)
            except Exception:
                desc = "callable"
        else:
            desc = str(obj)[:180]
        print(name, type(obj).__name__, desc)

print("\nRELEVANT HELPERS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and re.search(r"(post|prelude|recur|carrier|decode|readout|case|transport)", name, re.I):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 180; forwards 10. Run 179 shows factual-trained carrier decoding substantially attenuates physical-dose contrasts, especially insulin, and fails uniform scaling/stopping. No new edits. Inspecting exact posterior/transport implementation before a held-out multi-scale contrast calibration.\n")
```
