The last test improved the CPU approximation, but its remaining error is too large to trust for weight edits. I’ll re-read the architecture and earlier findings to recover the exact computation. Matching the forward tool’s baseline would confirm the reconstruction; continued discrepancies would refute it.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
_recover_notes = Path("/work/notes.md").read_text()
print(_recover_notes[:22000])
print("\n=== RELEVANT TENSORS ===")
for _recover_name in NAMES:
    if any(s in _recover_name.lower() for s in
           ("head", "last", "dose", "transition", "additive", "readout", "value")):
        print(_recover_name, SHAPES[_recover_name])
print("\n=== RETAINED VARIABLES ===")
for _recover_key, _recover_obj in list(globals().items()):
    if _recover_key.startswith("_") or _recover_key in ("NAMES", "SHAPES", "VOCAB"):
        continue
    if callable(_recover_obj):
        if any(s in _recover_key.lower() for s in ("head", "rd", "dose", "causal", "state", "target")):
            try:
                print(_recover_key, inspect.signature(_recover_obj))
            except Exception:
                print(_recover_key, type(_recover_obj).__name__)
    elif hasattr(_recover_obj, "shape"):
        print(_recover_key, type(_recover_obj).__name__, _recover_obj.shape)
    elif isinstance(_recover_obj, dict):
        print(_recover_key, "dict", list(_recover_obj.keys())[:16])
    elif isinstance(_recover_obj, (list, tuple)) and len(_recover_obj) < 25:
        print(_recover_key, type(_recover_obj).__name__, str(_recover_obj)[:350])
print("\n=== CURRENT EDITS ===", edited())
```
