The architecture confirms two drug pathways into the state: the action encoder and the latent posterior. The records also suggest the original rollouts substantially understate insulin’s effect, though those estimates still need auditing. The output was truncated before the retained-object inventory, so I’ll retrieve that compactly. This will show whether we can build a controlled edit without more speculative reconstruction.

```python
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch

snapshot42 = list(globals().items())
print("=== TABLES ===")
for key, obj in snapshot42:
    if isinstance(obj, pd.DataFrame):
        print(key, obj.shape, ",".join(map(str, obj.columns[:12])))
print("=== ARRAY INVENTORY ===")
print("; ".join(f"{key}:{tuple(obj.shape)}"
                for key, obj in snapshot42
                if isinstance(obj, (np.ndarray, torch.Tensor)) and not key.startswith("_")))
print("=== DICTIONARY INVENTORY ===")
print("; ".join(f"{key}:({len(obj)})[{','.join(map(str, list(obj)[:5]))}]"
                for key, obj in snapshot42
                if isinstance(obj, dict) and not key.startswith("_")
                and key not in {"SHAPES", "VOCAB", "__builtins__"}))
print("=== USER FUNCTIONS ===")
for key, obj in snapshot42:
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(key, inspect.signature(obj))
        except Exception:
            print(key)
print("=== SELECTED TENSORS ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_head", "z_prior", "z_in", "delta_in", "a_in",
                               "c_out", "b_in", "attr_heads.value", "last_numeric", "transition.value")):
        print(name, SHAPES[name])
print("=== NOTES: EARLY IMPLEMENTATION ===")
notes42 = Path("/work/notes.md").read_text()
start42 = notes42.find("## Run 10")
end42 = notes42.find("## Run 21")
print(notes42[start42:end42 if end42 > start42 else start42 + 4500][:4500])
print("=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 42: compact restart inventory\n"
            "Architecture recovered; retrieving retained statistical fits, encoding helpers, and states "
            "without repeating the full model listing. No edits. Python runs: 42; forward passes: 3.\n")
```
