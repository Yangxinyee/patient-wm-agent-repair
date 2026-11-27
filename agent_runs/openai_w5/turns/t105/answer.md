The last probes show that the direct action pathway can move both readings in the estimated directions. However, the required perturbations are large, so linear agreement alone is not enough. I’ll inspect the architecture and saved candidate construction before testing finite-dose changes. Stable rollout effects with negligible factual degradation would support this approach; saturation or a factual NLL increase would refute it.

```python
from pathlib import Path
import numpy as np, torch, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:14500])
print("\n=== KERNEL OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if inspect.isfunction(v):
        print("FUNCTION", k, "args", inspect.signature(v), "references", v.__code__.co_names[:35])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print("ARRAY", k, tuple(v.shape), str(v.dtype))
    elif isinstance(v, (dict, list, tuple)):
        print(type(v).__name__.upper(), k, "len", len(v), "keys/sample", list(v)[:8])
print("CURRENT EDITS", edited())
```
