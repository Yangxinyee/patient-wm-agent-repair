The earlier records suggest insulin lowers glucose substantially, unlike the original model’s small rollout response. The norepinephrine estimate is positive but less securely identified. The recovery output was truncated before the fitting objects appeared, so I’ll inspect those next. A useful correction must respond to dose changes without disrupting ordinary factual predictions.

```python
import inspect
import numpy as np
import pandas as pd
import torch

print("=== FUNCTIONS ===")
for key, val in list(globals().items()):
    if inspect.isfunction(val) and getattr(val, "__module__", "") == "__main__":
        try:
            print(key, str(inspect.signature(val)))
        except Exception:
            print(key)

print("=== DATAFRAMES ===")
for key, val in list(globals().items()):
    if isinstance(val, pd.DataFrame):
        print(key, val.shape, ",".join(map(str, val.columns[:35])))

print("=== LARGE ARRAYS ===")
for key, val in list(globals().items()):
    if isinstance(val, (np.ndarray, torch.Tensor)) and val.size if False else False:
        pass
    if isinstance(val, np.ndarray) and val.size > 100:
        print(key, val.shape, str(val.dtype))
    elif isinstance(val, torch.Tensor) and val.numel() > 100 and key not in {"obj", "val"}:
        print(key, tuple(val.shape), str(val.dtype))

print("=== DICTIONARIES / FITTED OBJECTS ===")
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"__builtins__", "SHAPES", "VOCAB"}:
        continue
    if isinstance(val, dict) and val:
        print(key, len(val), str(list(val.keys())[:8])[:180])
    elif hasattr(val, "predict") and not inspect.isclass(val):
        print(key, type(val).__name__)

print("=== ACTION / LATENT / VALUE TENSORS ===")
for key in NAMES:
    if any(s in key for s in ["encode_action", "z_post", "z_in", "z_head", "attr_heads.value", "embed.numeric"]):
        print(key, SHAPES[key])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 36: compact state recovery\n"
            "Early records imply insulin beta about -2.28 with persistence .857; original insulin x2 rollout -1.85 versus provisional record-based -13.59. NE residual-dose estimate +3.25 remains less securely identified. Recovering retained fitting objects without repeating architecture. No edits. Python runs: 36; forward passes: 3.\n")
```