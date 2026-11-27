The new norepinephrine gate detects nearly all treated bundles without firing on untreated ones. That supports using it as an input channel, but does not establish a drug effect. Next I’ll recover the exact channel construction and check its dose sensitivity before changing weights; weak sensitivity or leakage would reject this approach.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nCURRENT EDITS")
for name in edited():
    print(name, SHAPES[name])

print("\nRETAINED CHANNEL OBJECTS")
tokens181 = ("104", "117", "118", "123", "178", "179", "180")
items181 = list(globals().items())
for name, obj in items181:
    if name.startswith("_") or not any(t in name for t in tokens181):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:12])
    elif callable(obj) and hasattr(obj, "__code__"):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig, "dependencies:", obj.__code__.co_names)
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, "length", len(obj))

print("\nCAPTURE / ACTION / REPLAY INTERFACES")
for name, obj in items181:
    if callable(obj) and hasattr(obj, "__code__") and any(
        t in name.lower() for t in ("capture", "intervention_bank", "replay", "channel")
    ):
        try:
            print(name, inspect.signature(obj), obj.__code__.co_names)
        except Exception:
            pass

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 181\n"
        "Run-180 NE gate with ridge 100 had held-out accuracy .99979, "
        "active miss rate .00171 and no inactive firing. This establishes "
        "bundle discrimination only, not dose decoding or causal effect. "
        "Recovering architecture and exact retained channel interfaces "
        "before testing an independent NE channel. No weights changed.\n"
    )
```