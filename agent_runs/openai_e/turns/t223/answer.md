The old norepinephrine edit activates almost identically on original and insulin-repaired states, so state drift does not explain its factual cost. The dose contrasts are measurable, but posterior reconstruction is still imperfect. I’ll recover the exact edit construction before testing a correction that responds to dose changes while remaining small on factual records.

```python
from pathlib import Path
import re
import numpy as np
import torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRECORD SCHEMA")
print(Path("/data/README.md").read_text())

print("\nRELEVANT NOTEBOOK EVIDENCE")
notes193 = Path("/work/notes.md").read_text()
lines193 = notes193.splitlines()
patterns193 = [
    r"target", r"causal", r"beta", r"persistence",
    r"baseline77", r"candidate151", r"reduced107",
]
selected193 = [
    (i + 1, s) for i, s in enumerate(lines193)
    if any(re.search(p, s, re.I) for p in patterns193)
]
for i, s in selected193[-45:]:
    print(f"{i}: {s[:1000]}")

print("\nSAVED EDIT SCHEMAS")
for filename193 in [
    "baseline77.npz", "ne_candidate151.npz",
    "ne_reduced107.npz", "ne_detector81.npz",
]:
    with np.load("/work/" + filename193, allow_pickle=False) as archive193:
        print(filename193, [(k, archive193[k].shape) for k in archive193.files])

print("\nRECENT KERNEL OBJECTS")
snapshot193 = list(globals().items())
for name193, obj193 in snapshot193:
    if re.search(r"(190|191|192)$", name193):
        if isinstance(obj193, dict):
            print(name193, "dict", list(obj193.keys())[:25])
        elif hasattr(obj193, "shape"):
            print(name193, type(obj193).__name__, tuple(obj193.shape))
        elif callable(obj193):
            print(name193, "function")
        elif isinstance(obj193, (list, tuple)):
            print(name193, type(obj193).__name__, "length", len(obj193))
        else:
            print(name193, type(obj193).__name__, repr(obj193)[:200])

print("\nRELEVANT TENSOR SHAPES")
for name193 in NAMES:
    if any(s in name193 for s in ["b_in", "action", "world", "posterior"]):
        print(name193, SHAPES[name193])

with Path("/work/notes.md").open("a") as f193:
    f193.write(
        "\n## Run 193: recover exact contrast-edit construction\n"
        "Run192 refuted repaired-state anchor drift. Reading architecture and "
        "recovering saved NE constructions and contrast variables before a "
        "factual-null, dose-responsive auxiliary fit. Posterior offsets remain "
        "unverified; no weights changed. Python runs: 193; forwards: 22.\n"
    )
```