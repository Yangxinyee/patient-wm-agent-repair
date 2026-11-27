Scale 12 barely changed insulin x2 and made the norepinephrine responses asymmetric. That suggests numeric amplification alone cannot repair the model. I’ll inspect the architecture, earlier evidence and factual-fit result to find a more targeted pathway. A separate or saturating dose pathway would support this hypothesis; a large factual-fit loss would also rule out keeping scale 12.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

arch_text = Path("/model/ARCHITECTURE.md").read_text()
notes_text = Path("/work/notes.md").read_text()
print("=== ARCHITECTURE ===")
print(arch_text)
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
print("=== EARLIER NOTES ===")
print(notes_text[:16000])
print("=== FORWARD 8 FILES ===")
for p in sorted(Path("/work/fwd_8").iterdir()):
    print(p.name, p.stat().st_size)
    if p.suffix == ".json":
        print(p.read_text()[:6000])
print("=== KERNEL VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB", "notes_text", "arch_text"}:
        continue
    if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:12])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")
print("=== CURRENT EDITS ===")
print(edited())
```
