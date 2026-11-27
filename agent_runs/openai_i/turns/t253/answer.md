The records suggest opposite effects from the raw dose–reading associations: adjusted estimates are about −2 glucose units per insulin-rate unit and +3 MAP units per norepinephrine-rate unit. Those estimates still depend on adequate history adjustment. I’ll retrieve the previous validation results and the routing layout to see whether this candidate can deliver those responses without exceeding the factual-loss limit.

```python
import re, numpy as np
from pathlib import Path

notes226 = Path("/work/notes.md").read_text()
blocks226 = re.split(r"(?=^## )", notes226, flags=re.M)
print("PREVIOUS VALIDATION SUMMARIES")
for b in blocks226:
    lines = b.splitlines()
    hits = [x for x in lines[1:] if any(s in x.lower() for s in
            ["original rollouts", "edited rollouts", "dnll", "nll +",
             "nll from", "nll:", "nll =", "safeguard", "persistence",
             "candidate186", "fallback188"])]
    if hits:
        print(lines[0])
        for x in hits[:3]:
            print(x[:340])

print("\nCANDIDATE 186–190 NOTES")
for b in blocks226:
    if re.search(r"Python (186|187|188|189|190)\b", b[:150]):
        print(b[:1100])

print("\nSMALL ROUTE VARIABLES")
for key in sorted(list(globals())):
    if key.endswith(("220", "221", "222", "223")):
        obj = globals()[key]
        if callable(obj):
            print(key, "function", obj.__code__.co_varnames[:obj.__code__.co_argcount],
                  "uses", obj.__code__.co_names)
        elif isinstance(obj, dict):
            print(key, "dict", len(obj), list(obj)[:6])
        elif hasattr(obj, "shape"):
            print(key, tuple(obj.shape))
        elif isinstance(obj, (int, float, str)):
            print(key, repr(obj)[:100])

print("\nSAVED ROUTE TENSORS")
with np.load("/work/small_route222.npz") as f226:
    print([(k, f226[k].shape) for k in f226.files])
print("CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f226:
    f226.write("\n## Python 226 — compact validation and route recovery\n"
               "- Architecture and early history-adjusted drug estimates recovered.\n"
               "- Inspecting previous candidate validation and small-route tensors; no further weight changes.\n")
```