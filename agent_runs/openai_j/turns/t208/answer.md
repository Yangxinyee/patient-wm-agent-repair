The records support a substantial insulin effect, but the norepinephrine estimate relies on stronger assumptions about measured history; its sensor-difference instrument is weak. I’ll inspect the MAP candidate and its factual errors next. An extension must improve held-out fit as well as dose sensitivity—not merely amplify an uncertain target.

```python
import re
from pathlib import Path

print("RELEVANT FUNCTIONS")
fn187 = sorted(
    k for k, v in list(globals().items())
    if callable(v) and getattr(v, "__module__", "") == "__main__"
    and re.search(r"map|head|wm_|iv|kernel|sensor|candidate|eval|compress", k, re.I)
)
print(fn187)

print("\nCANDIDATE VARIABLE INVENTORY")
for k, v in sorted(list(globals().items())):
    if not re.match(r"^(map164|map165|map17|head177|mc169|wm_weights145|wm_.*145|best_)", k):
        continue
    if isinstance(v, dict):
        print(k, "dict", list(v.keys())[:18])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v), repr(v[:4])[:150])
    elif isinstance(v, (int, float, str)):
        print(k, repr(v)[:150])

print("\nKERNEL TARGETS")
print(Path("/work/kernel_targets.csv").read_text()[:1500])
print("\nSENSOR IV")
print(Path("/work/sensor_iv.csv").read_text()[:1500])

sections187 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("\nLATER CAUSAL AUDITS")
for s in sections187:
    title = s.split("\n", 1)[0]
    if re.search(r"Fixed-dose response|Run 182|Run 183|Run 184|Run 185", title, re.I):
        print(s[:1800])
```
