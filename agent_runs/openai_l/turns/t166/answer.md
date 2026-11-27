The dose wire looks approximately linear, and the controller can reproduce the fitted decay rates, but stronger regularization trades accuracy for less factual leakage. Neither component has yet been validated as a complete edit. I’ll recover their exact interfaces before assembling them; the edit must carry dose changes forward without materially changing factual predictions.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in ["b_in", "c_out", "prelude_norm", "value", "flow"]):
        print(name, SHAPES[name])

print("\n=== RECOVERED COMPONENTS ===")
for name in sorted(list(globals())):
    if name.endswith(("118", "125", "145", "152", "153")):
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, "norm", round(float(np.linalg.norm(obj)), 5))
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:30])
        elif callable(obj):
            print(name, "callable")
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj),
                  "preview", repr(obj)[:350])
        else:
            print(name, type(obj).__name__, repr(obj)[:350])

print("\n=== CAUSAL-EVIDENCE NOTE EXCERPTS ===")
lines = Path("/work/notes.md").read_text().splitlines()
selected = [i for i, line in enumerate(lines)
            if any(s in line.lower() for s in
                   ["eight-lag", "kernel", "discontinuity", "accepted immediate",
                    "provisional hour", "original rollout", "rd estimate"])]
for i in selected[-18:]:
    print("\n".join(lines[max(0, i-1):min(len(lines), i+3)]))
```
