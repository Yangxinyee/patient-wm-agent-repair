The edit preserves factual fit but barely changes the dose responses. That rejects it as an effective repair. I’ll recover the implementation and calibration details to distinguish an inactive dose feature from a latent correction that is simply too weak.

```python
from pathlib import Path
import numpy as np
import torch

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 128; forwards 7–8 — first candidate ineffective\n"
            "- Factual tuning NLL: original 1.63155, edited 1.63156 (100 stays).\n"
            "- Matched rollouts (/work/fwd_9): insulin x2 -2.016 -> -1.985; half 1.051 -> 1.015; "
            "NE x2 .309 -> .345; half -.070 -> -.032.\n"
            "- Factual fit passes locally, but dose correction is essentially ineffective; do not treat this as a successful repair.\n"
            "- Recovering exact architecture, feature gates, and latent-direction calibration before further edits.\n")

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
print("=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:11000])
print("=== RETAINED FUNCTIONS ===")
_g128 = list(globals().items())
print([k for k,v in _g128 if callable(v) and not k.startswith("_")])
print("=== RELEVANT RETAINED OBJECTS ===")
_terms128 = ("post", "dose", "rate", "gate", "policy", "dir", "innov", "candidate",
             "latent", "action", "comp", "replace", "gain", "decoder", "hidden")
for k,v in _g128:
    if k.startswith("_") or not any(t in k.lower() for t in _terms128):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", "keys", list(v.keys())[:25])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "len", len(v), "preview", repr(v[:4])[:220])
    elif isinstance(v, (str, int, float, bool, np.number)):
        print(k, repr(v)[:250])
```
