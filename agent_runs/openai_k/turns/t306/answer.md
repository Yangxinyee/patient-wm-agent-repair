The architecture confirms that `core.b_in.weight` can change how dose information enters the recurrent state. The output was truncated before the fit inventory, so I’ll retrieve only the essential objects. The next candidate must improve MAP responses, not merely its local routing cost.

```python
import inspect
import numpy as np
import torch
import pandas as pd
from pathlib import Path

r270_items = [(k, v) for k, v in list(globals().items())
              if not k.startswith(("_", "r269_", "r270_"))]
print("=== RECENT OBJECTS ===")
for r270_k, r270_v in r270_items[-85:]:
    if isinstance(r270_v, (np.ndarray, torch.Tensor)):
        print(r270_k, type(r270_v).__name__, tuple(r270_v.shape))
    elif inspect.isfunction(r270_v):
        print(r270_k, "function", inspect.signature(r270_v))
    elif isinstance(r270_v, dict):
        print(r270_k, "dict", list(r270_v.keys())[:8])
    elif isinstance(r270_v, (tuple, list)):
        print(r270_k, type(r270_v).__name__, len(r270_v))
    elif isinstance(r270_v, (str, int, float, np.number)):
        print(r270_k, repr(r270_v)[:100])
    else:
        print(r270_k, type(r270_v).__name__)

print("=== USER FUNCTION NAMES ===")
print([k for k, v in r270_items
       if inspect.isfunction(v) and v.__module__ == "__main__"])

print("=== LAST SIX FUNCTION DEFINITIONS ===")
r270_funcs = [(k, v) for k, v in r270_items
              if inspect.isfunction(v) and v.__module__ == "__main__"]
for r270_k, r270_v in r270_funcs[-6:]:
    try:
        print(r270_k, inspect.getsource(r270_v)[:700])
    except (OSError, TypeError):
        print(r270_k, "source unavailable",
              r270_v.__code__.co_names)

print("=== CANDIDATE ARCHIVES ===")
for r270_fn in ("glucose_115_candidate.npz",
                "recurrent_glucose_valid.npz",
                "rejected_map_trial.npz"):
    with np.load("/work/" + r270_fn) as r270_z:
        print(r270_fn, [(k, r270_z[k].shape) for k in r270_z.files])

print("=== FACTUAL VALIDATION BATCH ===")
r270_df = pd.read_csv("/work/fwd_24/factual_edited.csv")
print("columns", r270_df.columns.tolist())
if "stay_id" in r270_df.columns:
    print("stays", r270_df.stay_id.drop_duplicates().tolist())
print("current edits", edited())

with open("/work/notes.md", "a") as r270_f:
    r270_f.write(
        "\n## Run 270: bounded interface recovery\n"
        "- Run 269 output truncated before fit inventory; retrieving compact objects and archives.\n"
        "- No weights changed. Python runs: 270; forward passes: 24.\n"
    )
```