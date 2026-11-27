The reconstruction errors are concentrated in reading and visit bundles; drug-only bundles are much closer. That points to an encoding mismatch, not evidence for a weight change. I’ll recover the architecture and retained analysis helpers before proceeding. Exact agreement with the exposed states would validate the reconstruction; persistent discrepancies would refute it.

```python
from pathlib import Path
import types
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
print(sorted(k for k, v in list(globals().items())
             if isinstance(v, types.FunctionType) and not k.startswith("_")))

print("\n=== RETAINED DATA / STATE OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, list(v.columns[:12]))
    elif isinstance(v, dict) and any(s in k.lower() for s in
                                    ("state", "effect", "causal", "fit", "helper", "bundle")):
        print(k, "dict keys:", list(v)[:18])

print("\n=== EARLY NOTEBOOK: RECORD-BASED IDENTIFICATION ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:8500])
print("\nCurrent edits:", edited())
```
