The earlier analysis found insulin’s raw association had the wrong sign after accounting for dosing policy, and norepinephrine also required policy adjustment. The original rollouts barely reflected the inferred effects. I’ll now recover the saved numerical targets and nonlinear edit coefficients; a useful trial must move toward those targets without exceeding the factual-fit allowance.

```python
import re
from pathlib import Path
import numpy as np

_notes = Path("/work/notes.md").read_text()
_lines = [s for s in _notes.splitlines()
          if re.search(r"target|truth|beta=|persistence=|causal estimate|record-based|true effect",
                       s, re.I)]
print("=== TARGET / IDENTIFICATION NOTES (last relevant lines) ===")
print("\n".join(_lines[-28:])[:3200])

print("=== SAVED CONTRACTS ===")
for _fn in ["joint236.npz", "ne_nonlinear250.npz", "prelude_direction255.npz"]:
    with np.load("/work/" + _fn) as _a:
        print(_fn)
        for _k in _a.files:
            _v = _a[_k]
            print(_k, _v.shape, _v.dtype,
                  _v.tolist() if _v.size <= 10 else
                  ("norm=" + str(float(np.linalg.norm(_v))))
                  if np.issubdtype(_v.dtype, np.number) else "")

print("=== HELPER NAMES ===")
print([k for k, v in list(globals().items())
       if not k.startswith("_") and callable(v)])
print("=== RELEVANT ARRAY NAMES ===")
print([(k, type(v).__name__, getattr(v, "shape", None))
       for k, v in list(globals().items())
       if not k.startswith("_") and
       any(s in k.lower() for s in ["244", "250", "255", "258", "ids", "target", "truth"])])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 260: bounded recovery\n"
             "Architecture and initial record-based identification recovered. "
             "Printing compact target evidence and exact saved coefficients after previous output truncation. "
             "No weights changed. Python runs: 260; forwards: 29.\n")
```