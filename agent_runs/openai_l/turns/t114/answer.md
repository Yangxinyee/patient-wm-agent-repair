The raw posterior inputs preserve dose changes much better than the hidden features, while varying relatively little on factual records. That makes an upstream correction worth testing, but it does not yet prove that a weight edit will preserve predictions. I’ll recover the architecture and retained functions first; subsequent tests must show both selective dose sensitivity and unchanged factual fit.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes105 = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE ===")
print(notes105[:10000])

print("\n=== RETAINED FUNCTIONS ===")
for key105, val105 in list(globals().items()):
    if inspect.isfunction(val105) and getattr(val105, "__module__", "") == "__main__":
        try:
            sig105 = str(inspect.signature(val105))
        except Exception:
            sig105 = "(unknown)"
        print(key105, sig105)

print("\n=== RECENT VARIABLES ===")
for key105, val105 in list(globals().items()):
    if key105.endswith(("99", "100", "101", "102", "103", "104")):
        if hasattr(val105, "shape"):
            print(key105, tuple(val105.shape), getattr(val105, "dtype", ""))
        elif isinstance(val105, dict):
            print(key105, "dict keys:", list(val105)[:30])
        elif isinstance(val105, (str, float, int, list, tuple)):
            print(key105, repr(val105)[:700])

print("\n=== POSTERIOR / ACTION / RECURRENCE TENSORS ===")
for name105 in NAMES:
    if any(part105 in name105.lower() for part105 in
           ("posterior", "action", "prelude", "bypass", "gru", "transition")):
        print(name105, SHAPES[name105])

with open("/work/notes.md", "a") as f105:
    f105.write(
        "\n## Run 105 / restart recovery\n"
        "Raw posterior-input residuals preserve dose contrasts much better than hidden "
        "posterior features. Recovering architecture, causal evidence, and interfaces "
        "before constructing a selective upstream edit. No weights edited.\n"
    )
```
