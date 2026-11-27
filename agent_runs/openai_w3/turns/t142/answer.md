The last result confirms that the model is still unchanged. Dose-extraction helpers are available, but the local readout is only approximate, so it cannot establish factual safety. I’ll recover the architecture and edit helpers before a reversible pilot; the GPU tests must then show stronger dose responses without exceeding the factual-loss limit.

```python
from pathlib import Path
import numpy as np
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes129 = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(notes129[:14000])

print("\n=== RELEVANT TENSORS ===")
for name129 in NAMES:
    if any(s in name129 for s in ("prelude", "posterior", "prior", "readout", "decoder", "action", "z_in")):
        print(name129, SHAPES[name129])

print("\n=== RETAINED FUNCTIONS ===")
for name129, obj129 in list(globals().items()):
    if inspect.isfunction(obj129) and not name129.startswith("_"):
        code129 = getattr(obj129, "__code__", None)
        print(name129, inspect.signature(obj129),
              "calls:", code129.co_names if code129 else None)

print("\n=== EDITS ===", edited())
```
