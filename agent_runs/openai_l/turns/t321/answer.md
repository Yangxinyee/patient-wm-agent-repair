The reweighted heads look promising offline, especially for half-dose norepinephrine, but their rollout effects remain untested. The large baseline reconstruction discrepancy also needs checking. I’ll recover the fitting objects and records-derived targets before saving another candidate; agreement with the intended regression would support proceeding.

```python
from pathlib import Path
import numpy as np
import torch

arch282 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch282[:12000])
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text()[:5000])

notes282 = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION EVIDENCE ===")
print(notes282[:11000])

def describe282(x, depth=0):
    if torch.is_tensor(x):
        return f"Tensor{tuple(x.shape)} {x.dtype}"
    if isinstance(x, np.ndarray):
        return f"ndarray{x.shape} {x.dtype}"
    if isinstance(x, dict):
        keys = list(x)
        if depth >= 1:
            return {"type": "dict", "length": len(x), "keys": keys[:12]}
        return {str(k): describe282(x[k], depth + 1) for k in keys[:8]}
    if isinstance(x, (list, tuple)):
        return {
            "type": type(x).__name__,
            "length": len(x),
            "first": [describe282(v, depth + 1) for v in x[:5]]
        }
    if callable(x):
        return f"callable {getattr(x, '__name__', '')}"
    return repr(x)[:350]

print("=== LIVE FITTING OBJECTS ===")
for name in ["weighted281", "grid269", "arch263", "backup281", "T0",
             "GF", "RF", "GDI", "GDN", "RDI", "RDN",
             "feature_scale", "nll228", "calgrid279"]:
    if name in globals():
        print(name, describe282(globals()[name]))

if isinstance(globals().get("weighted281"), dict):
    for key in [(0.3, 0.25, 8.0), (0.25, 0.25, 8.0)]:
        if key in weighted281:
            print("CANDIDATE", key, describe282(weighted281[key]))

print("=== RELEVANT VARIABLE NAMES ===")
print(sorted(n for n in list(globals())
             if any(s in n.lower() for s in
                    ["281", "269", "263", "paired", "target", "truth", "effect", "contrast"])))
print("CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 282 / restart recovery\n"
            "No weights changed. Recovering architecture, records-derived causal evidence, "
            "and weighted281 candidate structure; baseline reconstruction discrepancy "
            "must be understood before staging a reweighted head.\n")
```