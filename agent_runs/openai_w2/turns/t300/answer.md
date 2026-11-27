The last run confirmed that no new rollout result exists; the stronger edit is still provisional. I’ll recover the record-based effect estimates and matched rollout cohort before using the three remaining forward passes. The stronger edit is worthwhile only if it improves agreement with those estimates without sacrificing factual-fit safety.

```python
from pathlib import Path
import re
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes242 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK: RECORD-BASED EVIDENCE")
print(notes242[:18000])

print("\nPERSISTENT VARIABLE INVENTORY")
keys242 = sorted([
    k for k in list(globals())
    if not k.startswith("_")
    and re.search(r"stay|rollout|dose_ids|target|causal|slope|persist|237|240", k, re.I)
])
for k in keys242:
    v = globals()[k]
    if hasattr(v, "shape"):
        print(k, type(v).__name__, "shape=", tuple(v.shape))
    elif isinstance(v, (list, tuple, set)):
        vv = list(v)
        print(k, type(v).__name__, "length=", len(vv), "first=", vv[:5])
    elif isinstance(v, dict):
        print(k, "dict keys=", list(v)[:15])
    elif isinstance(v, (str, int, float, np.number)):
        print(k, repr(v)[:250])

print("\nEDIT STATUS")
print(edited())
print("Current matches candidate_B237:",
      np.array_equal(np.asarray(load("core.b_in.weight")),
                     np.asarray(candidate_B237)))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run242: final-decision evidence recovery\n"
        "No weights changed. Forward48 has not run. Recovering architecture, "
        "record-based causal estimates, and the matched rollout cohort before "
        "spending the three remaining forward passes. candidate_B237 remains "
        "provisional; accepted fallback_B237 remains available.\n"
    )
```