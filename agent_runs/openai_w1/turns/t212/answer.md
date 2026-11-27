The matched states show that fivefold dose-input scaling reaches the value head, unlike the ineffective posterior edit. This is a promising pathway, but its usefulness still depends on separating dose effects from ordinary factual variation. I’ll recover the exact computation and cached analyses next. A dose-sensitive direction with little factual impact would support a targeted edit; substantial factual overlap would argue against it.

```python
from pathlib import Path
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nLIVE HELPERS")
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, getattr(getattr(v, "__code__", None), "co_varnames", ())[:12])

print("\nRELEVANT CACHES")
for k, v in list(globals().items()):
    if k.startswith(("r188", "r187", "rr_", "rp_", "ds_", "r168", "r164", "head", "causal", "kernel", "lag", "cf_", "train", "tuning")):
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        elif hasattr(v, "shape"):
            print(k, tuple(v.shape), type(v).__name__)
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:25])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
        else:
            print(k, type(v).__name__)

print("\nEARLY NOTEBOOK")
notes = Path("/work/notes.md").read_text()
print(notes[:16000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 189\nMatched original/5x states and value-head inputs are aligned. Recovering exact architecture, usable helpers, and early identification evidence before choosing a direct, factual-preserving dose pathway. No weights changed.\n")
```
