```python
import os, re, inspect, numpy as np
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
notes216 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK / CAUSAL EVIDENCE ===")
print(notes216[:18000])
print("\n=== CURRENT EDITS ===", edited())

print("\n=== RECENT KERNEL OBJECTS ===")
for n in sorted(list(globals())):
    if re.search(r"(19[0-9]|20[0-9]|21[0-5]|kernels19|targets179|readout179|R118|P125|U167|B167|wire_out152)", n):
        v = globals()[n]
        if isinstance(v, np.ndarray):
            print(n, "array", v.shape, str(v.dtype))
        elif isinstance(v, dict):
            print(n, "dict keys:", list(v)[:25])
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, "length", len(v))
        elif callable(v):
            try:
                print(n, "function", inspect.signature(v))
                if re.search(r"21[0-5]", n):
                    print(inspect.getsource(v)[:10000])
            except Exception:
                print(n, "function; source unavailable")
        else:
            print(n, type(v).__name__, repr(v)[:250])
print("\n=== AVAILABLE HELPER FUNCTIONS ===")
print([n for n, v in list(globals().items())
       if callable(v) and not n.startswith("_")
       and getattr(v, "__module__", "") in ("__main__", None)])
```
