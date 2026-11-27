The branch barely changed the rollouts: insulin x2 moved by another −0.038 and norepinephrine x2 by +0.022. It preserves factual fit but has not produced a meaningful repair. I’ll inspect the architecture and recover the branch construction before changing its strength. A measurable, dose-dependent response would support this pathway; continued insensitivity would argue for a different one.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECENT BRANCH CACHES ===")
_items177 = list(globals().items())
for _k, _v in sorted(_items177):
    if _k.startswith(("rp_", "r174_", "r175_", "r176_")):
        if hasattr(_v, "shape"):
            print(_k, type(_v).__name__, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_k, "dict keys:", list(_v)[:20])
        elif isinstance(_v, (list, tuple)):
            print(_k, type(_v).__name__, "len", len(_v), "sample", repr(_v[:8])[:400])
        elif isinstance(_v, (int, float, str, bool)):
            print(_k, repr(_v)[:400])
        elif callable(_v):
            print(_k, "callable", getattr(getattr(_v, "__code__", None), "co_varnames", ())[:16])
print("\n=== RELEVANT HELPERS ===")
for _k, _v in sorted(_items177):
    if callable(_v) and hasattr(_v, "__code__") and any(
        s in _k.lower() for s in ("post", "branch", "head", "latent", "causal", "contrast")
    ):
        print(_k, _v.__code__.co_varnames[:16])
print("\nCURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Forward 12–13 / Run 177\n"
             "Posterior four-neuron branch gains 4/4: tuning factual NLL unchanged at reported "
             "precision (1.53139). Paired rollout shifts original -> candidate: insulin x2 "
             "-1.773 -> -1.811; half .816 -> .807; NE x2 .048 -> .070; half -.481 -> -.473. "
             "Branch is too weak; recovering exact construction and transmission pathway before "
             "a controlled gain or pathway test. Counts: 177 Python runs, 13 forwards.\n")
```
