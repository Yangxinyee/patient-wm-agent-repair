The latent-state results suggest a possible way to carry glucose and MAP effects, but the remaining encoder mismatch makes a posterior edit premature. The fivefold dose scaling is only a fallback; it barely changed the norepinephrine response. I’ll recover the exact weight layout and cached inputs next. A candidate needs both a stronger dose response and preserved factual fit.

```python
from pathlib import Path
import numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== WEIGHT LAYOUT: ACTION, POSTERIOR, RECURRENCE, OUTPUT ===")
for name in NAMES:
    if any(s in name.lower() for s in ("numeric", "action", "posterior", "prelude", "recur", "transition", "slot", "out.")):
        print(name, SHAPES[name])
print("=== FOCUSED LIVE CACHE INVENTORY ===")
_items = []
for _k, _v in list(globals().items()):
    if _k.startswith(("ds_", "rr_", "post_", "latent_", "head_", "causal_", "kernel_", "probe_", "paired_")):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            _items.append((_k, type(_v).__name__, tuple(_v.shape)))
        elif isinstance(_v, dict):
            _items.append((_k, "dict", list(_v.keys())[:8]))
        elif callable(_v):
            _items.append((_k, "function", list(_v.__code__.co_varnames[:_v.__code__.co_argcount])))
        elif isinstance(_v, (list, tuple)):
            _items.append((_k, type(_v).__name__, len(_v)))
print(*_items[:65], sep="\n")
print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 169\nRestart recovery focused on exact architecture and causal-path caches. Fivefold numeric scaling remains provisional; posterior reconstruction is not yet exact, so no posterior edit is justified.\n")
``` 
