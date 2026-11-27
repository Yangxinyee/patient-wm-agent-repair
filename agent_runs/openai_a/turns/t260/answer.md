The intercept screen suggests a small factual-fit improvement, but it also weakens the insulin contrast. Since CPU contrasts have mispredicted rollouts before, this is not yet a better edit. I’ll recover the causal evidence and current implementation before choosing another reversible trial; it must improve the actual rollout while retaining a factual-fit margin.

```python
from pathlib import Path
import re, numpy as np, pandas as pd, torch

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README\n", Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\nEARLY EVIDENCE\n", _notes[:16000])

print("\nRELEVANT KERNEL OBJECTS")
_pat = re.compile(r"cache|r3|contrast|causal|target|checkpoint|policy|basis|finger|decoder|ridge|gate|install|ids|rd_|beta|persist|head|screen", re.I)
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or not _pat.search(_name):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", len(_obj), "keys:", list(_obj)[:12])
    elif callable(_obj):
        print(_name, "callable")
    elif isinstance(_obj, (float, int, str, list, tuple)):
        print(_name, repr(_obj)[:220])

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Restart recovery; Python 225, forwards 26/50\n"
             "- Last intercept screen was a CPU proxy only; no candidate accepted.\n"
             "- Recovering architecture, record-derived identification, and current helpers.\n"
             "- stronger6_checkpoint remains the protected GPU-validated fallback.\n")
```