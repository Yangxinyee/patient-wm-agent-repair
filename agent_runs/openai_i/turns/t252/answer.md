The zero-response routing edit increased factual NLL by 0.00509, just beyond the allowed 0.005. It is not safe to submit. I’ll recover the earlier effect estimates and routing details before choosing a smaller edit; any candidate must demonstrate both a useful dose response and preserved factual fit.

```python
from pathlib import Path
import re, numpy as np, torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())

notes225 = Path("/work/notes.md").read_text()
blocks225 = re.split(r"(?=^## )", notes225, flags=re.M)
print("\nEARLY EVIDENCE")
print("".join(blocks225[:12])[:16000])
print("\nPRIOR VALIDATION / CANDIDATE RESULTS")
selected225 = []
for b in blocks225[12:]:
    if any(s in b.lower() for s in [
        "forward 7", "forward 8", "forward 9", "forward 10", "forward 11",
        "forward 12", "forward 13", "forward 14", "forward 15", "forward 16",
        "fallback", "candidate186", "response coefficients", "true effect",
        "placebo", "persistence", "instrument"
    ]):
        selected225.append(b)
print("".join(selected225)[-19000:])

print("\nROUTING VARIABLES")
for key in sorted(list(globals())):
    if key.endswith(("220", "221", "222", "223")):
        obj = globals()[key]
        if callable(obj):
            print(key, "function", obj.__code__.co_varnames[:obj.__code__.co_argcount],
                  "globals", obj.__code__.co_names)
        elif isinstance(obj, dict):
            print(key, "dict", len(obj), list(obj)[:12])
        elif hasattr(obj, "shape"):
            print(key, tuple(obj.shape), str(getattr(obj, "dtype", "")))
        elif isinstance(obj, (int, float, str)):
            print(key, repr(obj)[:200])
print("\nCURRENT EDITS", edited())
print("\nSAVED SMALL ROUTE")
with np.load("/work/small_route222.npz") as f225:
    print([(k, f225[k].shape) for k in f225.files])

with open("/work/notes.md", "a") as f225:
    f225.write("\n## Python 225 — recover evidence and small-route geometry\n"
               "- Current neutral small route failed first-100 factual safeguard by +.00509.\n"
               "- Recovering architecture, earlier causal evidence, and candidate contracts before a response edit.\n"
               "- No additional tensors changed; this candidate is not approved for submission.\n")
```