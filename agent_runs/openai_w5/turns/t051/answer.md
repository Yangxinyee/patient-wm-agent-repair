The architecture confirms that doses influence readings through the recurrent state, while separate reading-register paths support factual prediction. The inventory stopped because it mistook a module for an array; no weights changed. I’ll safely inspect the saved analysis objects and tensor shapes to identify a testable dose pathway.

```python
import builtins
import types
import numpy as np
import torch
import pandas as pd

print("--- Persistent objects ---")
for name, obj in builtins.list(globals().items()):
    if name.startswith("_") or name in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, types.FunctionType):
        args = obj.__code__.co_varnames[:obj.__code__.co_argcount]
        print(name, "function", args)
    elif isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        print(name, "dict", builtins.list(obj.keys())[:15])
    elif isinstance(obj, (builtins.list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj)[:160])

print("\n--- Tensor names and shapes ---")
for name in NAMES:
    print(name, SHAPES[name])
print("\n--- Current edits ---", edited())
print("\n--- Vocabulary config ---", VOCAB["config"])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart, runs 42–43\n"
            "- Re-read architecture. Doses enter recurrent state; value heads also use sensor-specific last-reading registers.\n"
            "- Run 42 inventory failed on a module's callable shape attribute, before any edit.\n"
            "- Best local recurrence remains approximate; current edits are empty, 6 forwards used.\n")
```
