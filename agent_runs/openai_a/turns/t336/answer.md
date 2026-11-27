The latest 100-stay check increased factual NLL by 0.00445 nats, below the 0.005 limit for that subset. That is encouraging but not sufficient. I’ll recover the prior evidence and inspect all four validation batches; a pooled increase below the limit would support keeping this edit, while an increase above it would rule it out.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import json, inspect, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

_recovery_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK LENGTH ===", len(_recovery_notes))
print("=== EARLIER EVIDENCE ===")
print(_recovery_notes[:22000])
print("=== RECENT NOTEBOOK ===")
print(_recovery_notes[-8500:])

print("=== RELEVANT KERNEL OBJECTS ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and (
        callable(_obj) and getattr(_obj, "__module__", "") == "__main__"
        or re.search(r"checkpoint|sharp|causal|effect|activity|gate|target|panel|tuning|train|candidate", _name, re.I)
    ):
        _shape = getattr(_obj, "shape", None)
        _detail = f"shape={_shape}" if _shape is not None else (
            f"keys={list(_obj)[:16]}" if isinstance(_obj, dict)
            else f"length={len(_obj)}" if isinstance(_obj, (list, tuple)) else ""
        )
        print(_name, type(_obj).__name__, _detail)

print("=== CURRENT EDITS ===", edited())
for _k in (36, 37, 38, 39):
    _folder = Path(f"/work/fwd_{_k}")
    print("=== FORWARD", _k, "FILES ===", [p.name for p in _folder.iterdir()])
    for _model in ("original", "edited"):
        _file = _folder / f"factual_{_model}.csv"
        if _file.exists():
            _df = pd.read_csv(_file)
            _nllcols = [c for c in _df.columns if "nll" in c.lower()]
            print(_model, "shape", _df.shape, "columns", list(_df.columns),
                  "NLL means", {c: float(_df[c].mean()) for c in _nllcols})
            if _k == 36 and _model == "original":
                print(_df.head(2).to_string(index=False))
```
