The last run confirmed the data alignment, but the reconstructed head mixed incompatible bin dimensions. No weights changed. I’ll reread the architecture and inspect the head tensors before proceeding; this should identify the mismatch and a suitable place for the causal edit.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

print("\n=== TENSORS ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
print(type(VOCAB))
if isinstance(VOCAB, dict):
    for key, value in VOCAB.items():
        text = repr(value)
        print(key, text[:4000])

print("\n=== RETAINED HELPERS ===")
for name in ("cpu_head_parts", "cpu_head_probs", "repair_head_features"):
    fn = globals().get(name)
    if callable(fn):
        print(name, "args:", fn.__code__.co_varnames[:fn.__code__.co_argcount],
              "defaults:", fn.__defaults__,
              "references:", fn.__code__.co_names)

print("\n=== EARLY NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:18000])
print("\nCURRENT EDITS:", edited())
```
