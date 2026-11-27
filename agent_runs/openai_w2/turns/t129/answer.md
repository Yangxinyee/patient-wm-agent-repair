The fitted dose signals are cleaner for norepinephrine than for insulin, but that alone does not establish a safe weight edit. I’ll recover the architecture and saved helpers before choosing an injection point. Accurate reconstruction and a measurable dose response would support this route; failure on either would rule it out.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14500])
print("\n=== RELEVANT KERNEL VARIABLES ===")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in [
        "112", "111", "110", "prelude", "repair", "recur", "step",
        "probe", "ffn", "causal", "target", "bundle", "benchmark"
    ]):
        v = globals()[k]
        if isinstance(v, np.ndarray):
            print(k, "array", v.shape, v.dtype)
        elif callable(v):
            print(k, "callable", end=" ")
            try:
                print(inspect.signature(v))
            except Exception:
                print("")
        elif isinstance(v, dict):
            print(k, "dict keys", list(v)[:35])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "length", len(v),
                  "sample", repr(v[:2])[:200])
        else:
            print(k, type(v).__name__, repr(v)[:350])
print("\nCurrent edits:", edited())
```
