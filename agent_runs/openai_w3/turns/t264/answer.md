The corrected sensor order confirms the intended immediate drug responses, but the full dose-change pathway differs substantially from the simple correction. I’ll recover the installed implementation and record-derived targets before changing weights. A useful repair must improve both dose directions while keeping factual NLL within the limit.

```python
from pathlib import Path
import numpy as np
import torch
import inspect

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
notes236 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK\n", notes236[:16000])

g236 = dict(globals())
print("\nFUNCTIONS")
print(" ".join(sorted(k for k, v in g236.items()
                      if inspect.isfunction(v) and not k.startswith("_"))))
print("\nRECENT/CORE VARIABLES")
for k, v in sorted(g236.items()):
    if k.startswith("_") or not (
        any(s in k.lower() for s in
            ["193", "199", "206", "220", "221", "222", "228", "230",
             "231", "232", "234", "235", "target", "causal", "dose",
             "transport", "replay", "compens", "checkpoint", "sensor"])
    ):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:35])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif isinstance(v, (str, int, float, bool)):
        print(k, repr(v)[:240])
print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 236; forwards 16. Restart recovery: corrected sensor order does not explain the substantial mismatch between intended and actual paired-dose transport. Recovering architecture, identification notes, and exact installed implementation before making further edits. No weights changed.\n")
```