The channel-only edit preserved factual fit, and the candidate response directions lowered glucose approximately linearly without materially changing MAP. That supports testing the connection, but does not establish its sustained effect. I’ll recover the wiring and record-derived targets before adding it; rollout accuracy and the factual NLL limit will decide whether it stays.

```python
from pathlib import Path
import re
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED REPAIR OBJECTS ===")
_ns119 = dict(globals())
_pat119 = re.compile(
    r"channel|candidate|cand|ridge|jac|direction|bdir|bsol|response|"
    r"kernel|causal|innovation|gate|insulin|^J|^B|^dB|^vB|^bp|^b_",
    re.I,
)
for _name119, _obj119 in sorted(_ns119.items()):
    if _name119.startswith("_") or not _pat119.search(_name119):
        continue
    if isinstance(_obj119, (np.ndarray, torch.Tensor)):
        print(_name119, type(_obj119).__name__, tuple(_obj119.shape))
    elif isinstance(_obj119, dict):
        print(_name119, "dict", list(_obj119.keys())[:24])
    elif callable(_obj119):
        _code119 = getattr(_obj119, "__code__", None)
        print(_name119, "callable",
              _code119.co_varnames[:_code119.co_argcount] if _code119 else "")
    elif isinstance(_obj119, (str, int, float, list, tuple)):
        print(_name119, repr(_obj119)[:240])

print("\n=== EARLIER RECORD EVIDENCE ===")
_notes119 = Path("/work/notes.md").read_text()
print(_notes119[:11000])
print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as _f119:
    _f119.write(
        "\n## Run 119\n"
        "Run-118 response directions are approximately linear and glucose-selective. "
        "Recovering exact retained wiring and earlier record evidence before connecting "
        "the response; finite differences are not sustained causal validation. "
        "No additional weights edited.\n"
    )
```