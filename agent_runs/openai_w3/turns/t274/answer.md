The new edit barely changed the rollout effects: insulin doubling remains too strong, insulin halving too weak, and norepinephrine effects too small. This contradicts the intended gain adjustment, while factual fit worsened. I’ll revert to the validated checkpoint and inspect the implementation; the next hypothesis must explain why the fitted dose contrast scarcely affects the actual rollout.

```python
from pathlib import Path
import numpy as np
import torch

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
g244 = globals()
near244 = sorted(k for k in list(g244) if any(s in k for s in ("243", "checkpoint222", "checkpoint221", "candidate206", "dd193")))
print("\nRELEVANT VARIABLES")
for k in near244:
    v = g244[k]
    if isinstance(v, dict):
        print(k, "dict", len(v), "keys", list(v)[:15])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    else:
        print(k, type(v).__name__)

restored244 = False
cp244 = g244.get("checkpoint243")
if isinstance(cp244, dict):
    tensor_items244 = {k: v for k, v in cp244.items() if k in NAMES}
    if tensor_items244 and all(tuple(v.shape) == tuple(SHAPES[k]) for k, v in tensor_items244.items()):
        for k in list(edited()):
            if k not in tensor_items244:
                drop_edit(k)
        for k, v in tensor_items244.items():
            save_edit(k, v)
        restored244 = True
print("Restored verified checkpoint243:", restored244)

print("\nAVAILABLE REPLAY/FITTING FUNCTIONS")
for k in sorted(list(g244)):
    v = g244[k]
    if callable(v) and hasattr(v, "__code__") and any(s in k.lower() for s in
        ("prelude", "replay", "carrier", "decode", "install", "direction", "token", "fit", "output", "transport", "step")):
        print(k, "args", v.__code__.co_varnames[:v.__code__.co_argcount])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 244; forwards 18. Candidate243 failed factual budget: +0.00595 nats. "
            "Rollouts -19.054,+1.339,+1.817,-1.064 scarcely changed despite intended rank-two gain adjustment. "
            f"Verified checkpoint243 restoration: {restored244}. "
            "Recovering exact architecture and helpers; simple gain interpretation is not supported.\n")
```