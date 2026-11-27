Removing the posterior’s action input barely changed the rollout responses, so it does not look like a useful repair. I’ll undo it and recheck the architecture and saved analysis objects. Next I’ll look for a dose-sensitive pathway that can be calibrated without disturbing factual predictions.

```python
from pathlib import Path
import numpy as np

drop_edit("z_post.0.weight")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n--- Tensor names and shapes ---")
for name in NAMES:
    print(name, SHAPES[name])

print("\n--- Persistent analysis objects ---")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    obj = globals()[key]
    if isinstance(obj, np.ndarray):
        print(key, "ndarray", obj.shape, obj.dtype)
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(key, type(obj).__name__, obj.shape, list(obj.columns)[:45])
    elif isinstance(obj, dict):
        print(key, "dict keys", list(obj.keys())[:35])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key, "function")

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 6 / run 26: failed posterior-action ablation\n"
            "- Removing z_post.0.weight[:,384:576] barely changed any dose-edit response.\n"
            "- Reverted the ablation; no tensors currently edited.\n"
            "- This does not support posterior action conditioning as the main cause of weak responses.\n"
            "- Reinspecting direct dose/state pathways before choosing another intervention.\n")
print("\nCurrent edits:", edited())
```