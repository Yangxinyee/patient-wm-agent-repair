Left-edge binning matches the existing labels, but the reconstructed CPU predictions still disagree with the GPU. I therefore cannot trust local likelihood estimates for calibration. I’ll recover the architecture, causal evidence, and current edit objects. A documented reconstruction mismatch would support this diagnosis; agreement would instead point to a faulty calibration calculation.

```python
import os, json, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

notes = open("/work/notes.md").read()
print("=== EARLY IDENTIFICATION / EDIT NOTES ===")
print(notes[:18000])
print("=== NOTEBOOK SIZE ===", len(notes))

print("=== KERNEL OBJECTS ===")
for key in sorted(list(globals())):
    if key.startswith("__") or key in {"notes", "key"}:
        continue
    obj = globals()[key]
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(key, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(key, "dict", len(obj), "keys:", list(obj)[:12])
    elif isinstance(obj, (list, tuple)):
        print(key, type(obj).__name__, len(obj), str(obj[:8])[:350])
    elif callable(obj):
        print(key, "callable")
    elif isinstance(obj, (int, float, str)):
        print(key, repr(obj)[:250])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 234: restart and validation policy\n"
            "- Left-edge binning agrees with existing labels; boundary changes do not explain local/GPU disagreement.\n"
            "- Recovered CPU likelihood calculations remain unvalidated and will not determine edit acceptance.\n"
            "- Recovering identification evidence and edit structure; subsequent acceptance requires paired GPU factual and rollout tests.\n")
```
