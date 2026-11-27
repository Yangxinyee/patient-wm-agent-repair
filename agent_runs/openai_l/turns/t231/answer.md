The dose decoders respond consistently across multipliers, but still vary substantially on factual records. A stronger linear correction could therefore damage factual fit. I’ll recover the head interfaces before testing a nonlinear, sensor-selective correction. It would support this approach if dose sensitivity increases without moving factual predictions; otherwise I’ll reject it.

```python
from pathlib import Path
import numpy as np

print(Path("/model/ARCHITECTURE.md").read_text())
print("\nCURRENT EDITS:", edited())
print("\nRELEVANT KERNEL VARIABLES:")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in [
        "179", "185", "200", "201", "202", "203", "204", "205", "206", "207",
        "kernel19", "gate", "family", "raw", "contrast", "head", "audit",
        "feature", "ridge", "case", "decoder", "split"
    ]):
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, obj.dtype)
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:18])
        elif callable(obj):
            import inspect
            try:
                print(name, "callable", inspect.signature(obj))
            except Exception:
                print(name, "callable")
        else:
            text = str(obj)
            print(name, type(obj).__name__, text[:220])
print("\nVALUE-HEAD TENSORS:")
for name in NAMES:
    if "attr_heads.value_bin" in name:
        print(name, SHAPES[name])
```
