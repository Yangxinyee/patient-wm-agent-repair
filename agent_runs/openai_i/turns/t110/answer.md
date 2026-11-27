Ignoring END atoms did not fix the posterior mismatch, so those reconstructed features are not reliable enough for an edit. I’ll recover the architecture and retained causal estimates to choose a directly testable correction. Agreement with exposed states would support that pathway; another mismatch would rule it out.

```python
from pathlib import Path
import inspect
import numpy as np

arch101 = Path("/model/ARCHITECTURE.md").read_text()
notes101 = Path("/work/notes.md").read_text()
print("ARCHITECTURE\n", arch101)
print("\nEARLY NOTEBOOK\n", notes101[:9000])

print("\nRETAINED OBJECT INVENTORY")
for name101 in sorted(list(globals())):
    if name101.startswith("_") or not any(
        s in name101.lower() for s in
        ("panel", "train", "tuning", "state", "bundle", "case", "lag",
         "effect", "target", "ridge", "repair", "post", "dose", "policy", "impulse")
    ):
        continue
    obj101 = globals()[name101]
    if callable(obj101):
        try:
            print(name101, "function", inspect.signature(obj101))
        except Exception:
            pass
    elif hasattr(obj101, "shape"):
        print(name101, type(obj101).__name__, obj101.shape)
    elif isinstance(obj101, dict):
        print(name101, "dict", len(obj101), "keys", list(obj101)[:12])
    elif isinstance(obj101, (list, tuple)):
        print(name101, type(obj101).__name__, len(obj101))
    elif isinstance(obj101, (str, int, float, np.number)):
        print(name101, repr(obj101)[:180])

with Path("/work/notes.md").open("a") as f101:
    f101.write("\n## Python 101 — architecture and evidence recovery\n"
               "- Ignoring END did not fix posterior reconstruction; those features are not approved for editing.\n"
               "- Recovering architecture and retained record-based estimates before choosing a directly testable correction.\n"
               "- No weights changed.\n")
```
