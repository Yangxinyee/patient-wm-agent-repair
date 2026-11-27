The insulin decoder now generalizes across stays and individual latent draws; the norepinephrine decoder does not. That supports testing an insulin-only correction first, not forcing a norepinephrine edit. I’ll recover the exact head implementation and fitted objects. A useful correction must survive the actual forward tool and preserve factual NLL—not merely fit cached features.

```python
from pathlib import Path
import os, inspect, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("=== RELEVANT KERNEL OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith(("r97", "r98", "r99")) or any(s in k.lower() for s in
            ("head", "decoder", "gate", "causal", "dml", "target", "rollout")):
        v = globals()[k]
        if callable(v):
            try:
                desc = str(inspect.signature(v))
            except Exception:
                desc = ""
        elif hasattr(v, "shape"):
            desc = str(v.shape)
        elif isinstance(v, dict):
            desc = "keys=" + str(list(v.keys())[:25])
        elif isinstance(v, (list, tuple)):
            desc = "len=" + str(len(v)) + " sample=" + repr(v[:3])[:300]
        else:
            desc = repr(v)[:160]
        print(k, type(v).__name__, desc)

print("=== EARLY NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:14000])
print("=== FORWARD OUTPUT FILES ===")
for d in sorted(Path("/work").glob("fwd_*")):
    print(d.name, [p.name for p in d.iterdir()])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Run 100\n"
            "Original weights remain unchanged. Run 99 established insulin innovation "
            "held-out single-draw R2=0.446, but no useful NE decoder. Recovering exact "
            "head helpers, gates and causal estimates before an insulin-only controlled "
            "edit. Actual paired factual NLL and dose rollouts are required validation.\n")
```
