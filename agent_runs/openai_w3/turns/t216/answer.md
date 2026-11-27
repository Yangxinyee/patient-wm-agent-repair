The pilot produces meaningful dose effects, but insulin halving is too weak and doubled norepinephrine is too small. Factual NLL is close to the limit. I’ll recover the decoder and transport implementation before changing anything, to distinguish dose-decoding errors from downstream response errors.

```python
from pathlib import Path
import re, types, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

g194 = dict(globals())
print("\n=== RETAINED FUNCTION NAMES ===")
print([k for k, v in g194.items()
       if isinstance(v, types.FunctionType) and not k.startswith("_")])

print("\n=== RECENT / TRANSPORT OBJECTS ===")
pat194 = re.compile(r"(18[0-9]|19[0-3]|carrier|transport|checkpoint|zero169|post154|zin154|zb154|J171)")
for k in sorted(g194):
    if not pat194.search(k):
        continue
    v = g194[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", {
            str(kk): tuple(vv.shape) if hasattr(vv, "shape") else type(vv).__name__
            for kk, vv in list(v.items())[:18]
        })
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif isinstance(v, (int, float, str, bool)):
        print(k, repr(v)[:200])

print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 194; forwards 12. Restart recovery: pilot gives meaningful but asymmetric effects, with factual NLL +0.00456 near the limit. Recovering exact retained decoder and transport before distinguishing carrier error from downstream rollout feedback. No weights changed.\n")
```