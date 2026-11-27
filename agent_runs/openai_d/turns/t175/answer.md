The warm-up scan suggests that suppressing early corrections may reduce factual loss without weakening late corrections, but this remains unverified in actual rollouts. I’ll recover the architecture, causal estimates, and cached helpers before changing weights. A useful candidate must improve factual fit on separate stays while retaining the dose response.

```python
from pathlib import Path
import re, inspect, hashlib
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK: INITIAL FINDINGS ===")
print(_notes[:16000])
print("=== NOTEBOOK SECTION INDEX ===")
print("\n".join(line for line in _notes.splitlines() if line.startswith("#")))

print("=== RECENT CACHED VARIABLES ===")
for _name in sorted(list(globals())):
    if not re.match(r"r(?:11[0-9]|12[0-9]|13[0-9]|14[0-9]|15[0-9])", _name):
        continue
    _obj = globals()[_name]
    if hasattr(_obj, "shape"):
        print(_name, type(_obj).__name__, "shape", _obj.shape)
    elif isinstance(_obj, dict):
        print(_name, "dict keys", list(_obj)[:30])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, "length", len(_obj))
    else:
        print(_name, type(_obj).__name__)

print("=== AVAILABLE LOCAL HELPERS ===")
for _name, _obj in list(globals().items()):
    if inspect.isfunction(_obj) and getattr(_obj, "__module__", None) == "__main__":
        try:
            print(_name, inspect.signature(_obj))
        except (TypeError, ValueError):
            print(_name)

print("=== INSTALLED EDITS ===")
for _name in edited():
    _a = load(_name)
    if hasattr(_a, "detach"):
        _a = _a.detach().cpu().numpy()
    _a = np.asarray(_a, dtype=np.float32)
    print(_name, _a.shape, hashlib.sha256(_a.tobytes()).hexdigest()[:12],
          "finite", bool(np.isfinite(_a).all()))
```
