The architecture gives drug information a separate action path, while reading heads also have strong last-value shortcuts. The recovered causal estimates still indicate substantial effects. I’ll check which action-path edits were already tested; a promising remaining edit should preserve factual fit while strengthening the missing response.

```python
from pathlib import Path
import numpy as np

notes = Path("/work/notes.md").read_text()
print("NOTEBOOK CONTINUATION")
print(notes[5500:11000])

print("\nAVAILABLE ARRAYS AND HELPERS")
lines = []
for name, value in sorted(list(globals().items())):
    if name.startswith("_") or name in {"In", "Out", "value", "notes"}:
        continue
    if isinstance(value, np.ndarray):
        lines.append(f"{name}: {value.shape}")
    elif isinstance(value, dict) and any(s in name.lower() for s in
                                       ["state", "fit", "edit", "dir", "shear", "candidate"]):
        lines.append(f"{name}: keys {list(value)[:8]}")
    elif callable(value) and any(s in name.lower() for s in
                                ["fit", "shear", "transform", "install", "direction"]):
        import inspect
        try:
            lines.append(f"{name}{inspect.signature(value)}")
        except Exception:
            lines.append(name)
print("\n".join(lines)[:2300])

print("\nACTION-PATH TENSORS")
for name in NAMES:
    if name.startswith(("encode_action.", "z_post.", "last_numeric.")):
        print(name, SHAPES[name])
```
