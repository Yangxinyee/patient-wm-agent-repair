The architecture confirms that the value head can change reading predictions without directly changing the other heads. The early notes also show that the simple state-space estimates were rejected because they conflicted with threshold evidence. I’ll recover the accepted response kernels and recent candidate objects; the next test needs to use those record-supported targets, not the rejected estimates.

```python
import re, inspect, numpy as np
from pathlib import Path

text217 = Path("/work/notes.md").read_text()
print("=== RECORD-BASED EFFECT EVIDENCE ===")
sections217 = re.split(r"(?=^## )", text217, flags=re.M)
for section in sections217:
    if re.search(r"^## .*?(?:Run 1[3-9]\b|Run 2[0-5]\b|kernel|accepted causal)", section, re.I):
        print(section[:1400])
        if sum(len(s[:1400]) for s in sections217[:sections217.index(section)+1]
               if re.search(r"^## .*?(?:Run 1[3-9]\b|Run 2[0-5]\b|kernel|accepted causal)", s, re.I)) > 4500:
            break

print("\n=== LATEST OBJECTS ===")
for n in sorted(list(globals())):
    if re.search(r"(21[0-5]|kernels19|targets179|readout179)$", n):
        v = globals()[n]
        if isinstance(v, np.ndarray):
            print(n, v.shape, str(v.dtype))
        elif isinstance(v, dict):
            print(n, "dict:", [(str(k), getattr(x, "shape", type(x).__name__))
                              for k, x in list(v.items())[:16]])
        elif callable(v):
            try:
                print(n, "function", inspect.signature(v))
            except Exception:
                print(n, "function")
        else:
            print(n, type(v).__name__, repr(v)[:180])

print("\n=== RECENT HELPER NAMES ===")
print([n for n, v in list(globals().items())
       if callable(v) and re.search(r"(19[0-9]|20[0-9]|21[0-5])", n)])

if "kernels19" in globals():
    print("\n=== KERNELS19 ===")
    print(repr(kernels19)[:2200])
```
