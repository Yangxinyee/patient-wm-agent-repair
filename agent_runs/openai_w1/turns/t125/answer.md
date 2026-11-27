The records show a negative adjusted insulin association, while the original rollout response is small. That is useful evidence, but it does not yet establish the full response over time. I’ll recover the lag estimates and head structure to choose an edit that preserves factual predictions rather than merely forcing the expected signs.

```python
from pathlib import Path
import numpy as np, torch

_notes = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION AND EDIT EXPERIMENTS ===")
print(_notes[:7600])

print("\n=== HEAD AND ACTION/POSTERIOR TENSOR SHAPES ===")
for _n in NAMES:
    if any(_n.startswith(p) for p in
           ("attr_heads.value", "last_numeric", "encode_action", "z_post", "z_head", "z_in", "query_norm")):
        print(_n, SHAPES[_n])

print("\n=== SELECTED KERNEL OBJECTS ===")
for _n, _v in sorted(list(globals().items())):
    if _n.startswith(("rr_", "r88_", "r95_")):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_n, type(_v).__name__, tuple(_v.shape))
        elif callable(_v):
            print(_n, "function")
        elif isinstance(_v, dict):
            print(_n, "dict", len(_v), "keys", list(_v)[:5])
        elif isinstance(_v, (int, float, str)):
            print(_n, repr(_v)[:100])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 114\nRecovered early identification evidence and tensor layouts. "
             "No causal magnitudes inferred from signs alone; no weights edited. "
             "Counts: 114 Python runs, 6 forward passes.\n")
```
