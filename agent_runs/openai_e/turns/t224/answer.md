The architecture confirms that dose changes affect both the recurrent state and the readings fed back during rollout. The notebook also shows that the reduced norepinephrine edit passed the factual limit but undershot the inferred effect. The output was truncated before the construction details; I’ll retrieve those compactly, without changing weights.

```python
import re
import numpy as np
import torch
from pathlib import Path

def as_numpy194(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("EARLY RECORD-DERIVED EFFECT EVIDENCE")
text194 = Path("/work/notes.md").read_text().splitlines()
hits194 = [
    (i + 1, s) for i, s in enumerate(text194)
    if re.search(r"(target|causal|beta|persistence|identif)", s, re.I)
]
for i194, s194 in hits194[:16]:
    print(f"{i194}: {s194[:450]}")

print("\nSAVED CONSTRUCTIONS")
for file194 in [
    "baseline77.npz", "ne_candidate151.npz",
    "ne_reduced107.npz", "ne_detector81.npz",
]:
    with np.load("/work/" + file194, allow_pickle=False) as a194:
        print(file194, [(k, a194[k].shape) for k in a194.files])

print("\nRECENT VARIABLES")
recent194 = [
    (n, x) for n, x in list(globals().items())
    if re.search(r"(190|191|192)$", n)
]
for n194, x194 in recent194[:65]:
    if isinstance(x194, dict):
        print(n194, "dict", list(x194.keys())[:12])
    elif hasattr(x194, "shape"):
        print(n194, type(x194).__name__, tuple(x194.shape))
    elif callable(x194):
        print(n194, "function")
    elif isinstance(x194, (list, tuple)):
        print(n194, type(x194).__name__, "length", len(x194))
    else:
        print(n194, type(x194).__name__, repr(x194)[:100])
if len(recent194) > 65:
    print("REMAINING NAMES", [n for n, x in recent194[65:]])

print("\nCORE INPUT TENSOR")
print(SHAPES["core.b_in.weight"])
print("EDITED", edited())

with Path("/work/notes.md").open("a") as f194:
    f194.write(
        "\n## Run 194: bounded construction recovery\n"
        "Run193 output truncated before recent variables and archive schemas. "
        "Recovering those compactly plus early causal evidence; no weights "
        "changed. Python runs: 194; forwards: 22.\n"
    )
```