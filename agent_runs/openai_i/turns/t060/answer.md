```python
from pathlib import Path
import types
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes_text[:14000])

print("\n=== RETAINED OBJECTS ===")
snapshot = list(globals().items())
print("Functions:", [n for n, v in snapshot
                     if isinstance(v, types.FunctionType) and not n.startswith("_")])
for n, v in snapshot:
    if n.startswith("_") or n in {"NAMES", "SHAPES", "VOCAB", "snapshot"}:
        continue
    if isinstance(v, pd.DataFrame):
        print("FRAME", n, v.shape, list(v.columns)[:24])
    elif isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 2:
        print("ARRAY", n, tuple(v.shape))
    elif isinstance(v, dict) and any(
        token in n.lower() for token in
        ("repair", "latent", "causal", "effect", "impulse", "state", "lag", "probe", "tune")
    ):
        print("DICT", n, "length", len(v), "keys", list(v)[:20])

print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 53 — restart recovery\n"
            "- No edits. Recovering architecture, early identification evidence, and retained "
            "objects before computing a dose-specific recurrent correction.\n")
```