```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("=== EARLY FINDINGS ===")
print(_notes[:16000])
print("=== RECENT SAVED EDIT CONTRACTS ===")
for _fn in ["joint236.npz", "ne_nonlinear250.npz", "prelude_direction255.npz"]:
    with np.load("/work/" + _fn) as _a:
        print(_fn)
        for _k in _a.files:
            _v = _a[_k]
            print(_k, _v.shape, _v.dtype,
                  _v.tolist() if _v.size <= 16 else
                  {"norm": float(np.linalg.norm(_v))} if np.issubdtype(_v.dtype, np.number) else "")
print("=== AVAILABLE HELPERS / RELEVANT GLOBALS ===")
print([(k, type(v).__name__, getattr(v, "shape", None))
       for k, v in list(globals().items())
       if not k.startswith("_") and
       (callable(v) or any(s in k.lower() for s in
        ["244", "250", "255", "258", "truth", "effect", "target", "tuning", "joint", "dose"]))])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 259: restart and trial preparation\n"
             "Local prelude MAP direction is stable but has glucose leakage; no GPU validation yet. "
             "Recovering architecture, causal evidence and saved edit contracts before deciding on a trial. "
             "Joint236 remains unchanged. Python runs: 259; forwards: 29.\n")
```