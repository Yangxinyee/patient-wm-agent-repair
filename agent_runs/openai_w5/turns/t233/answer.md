The second 100-stay sample shows a factual NLL increase of 0.00140, within the 0.005 limit. That supports factual preservation, but not causal accuracy. I’ll recover the architecture and record-based estimates before investigating the doubling/halving asymmetry; proportional dose transport would support the current design.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD-IDENTIFICATION NOTES ===")
print(_notes[:16000])
print("\n=== NOTEBOOK SECTION HEADINGS ===")
print("\n".join(x for x in _notes.splitlines() if x.startswith("#")))

print("\n=== PERSISTENT VARIABLES ===")
for _k, _v in list(globals().items()):
    if _k.startswith("_") or _k in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(_k, type(_v).__name__, tuple(_v.shape))
    elif isinstance(_v, dict):
        print(_k, "dict", len(_v), "keys:", list(_v)[:12])
    elif isinstance(_v, (list, tuple)):
        print(_k, type(_v).__name__, len(_v))
    elif callable(_v):
        print(_k, "callable")
    elif isinstance(_v, (str, int, float, bool)):
        print(_k, repr(_v)[:200])
print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart after forward 18 / run 206\n"
             "- Tuning B factual value NLL: 1.62038 versus original 1.61898, +0.00140.\n"
             "- Factual preservation holds on this sample; finite-dose causal asymmetry remains unresolved.\n"
             "- Recovering architecture, record-identification evidence, and kernel objects before further edits.\n")
```