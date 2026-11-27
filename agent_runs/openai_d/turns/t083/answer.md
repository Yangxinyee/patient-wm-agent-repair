The query is stable across latent draws, but the proposed reading-type gate does not separate the four readings. I’ll recover the architecture and earlier causal estimates before choosing another pathway. A valid correction must target the intended reading and preserve factual fit; the failed gate cannot establish that.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes_text[:14000])
print("\n=== PERSISTENT HELPERS ===")
for name, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)
print("\n=== RELEVANT ARRAY SHAPES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and any(
        s in name.lower() for s in ("wr_", "query", "head", "gate", "state", "coef")
    ):
        print(name, tuple(obj.shape))
print("\nCurrent edits:", edited())
```
