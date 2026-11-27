The 0.97 logit scaling helped only slightly: factual NLL is still **0.00594 above the original**, so this candidate fails the constraint. I’ll recover the causal estimates and exact edit structure before choosing a smaller, reversible adjustment. It must pass the factual limit and improve the dose responses—not merely increase their magnitude.

```python
import os, re, numpy as np

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD FORMAT ===")
print(open("/data/README.md").read())

notes283 = open("/work/notes.md").read()
print("=== EARLIER IDENTIFICATION AND EDIT HISTORY ===")
print(notes283[:18000])

print("=== RECOVERED KERNEL OBJECTS ===")
snapshot283 = list(globals().items())
for name, obj in snapshot283:
    if name.startswith("_") or name in {"In", "Out", "get_ipython", "notes283", "snapshot283"}:
        continue
    if re.search(r"target|causal|effect|coef|fit|base|trial|center|strength|tau|head|transport|tuning|ids|pair|private|checkpoint", name, re.I):
        shape = getattr(obj, "shape", None)
        if isinstance(obj, (str, int, float, np.number)):
            detail = repr(obj)[:220]
        elif isinstance(obj, dict):
            detail = "keys=" + repr(list(obj.keys())[:35])
        elif isinstance(obj, (list, tuple)):
            detail = "len=" + str(len(obj)) + "; first=" + repr(obj[:8])[:220]
        else:
            detail = "shape=" + str(shape)
        print(name, type(obj).__name__, detail)

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 29 / run 283: temperature trial insufficient\n"
            "- Scaling all value logits by 0.97 gives tuning-A NLL 1.64527 vs 1.63933, +0.00594: still fails.\n"
            "- No weights changed in this recovery run. Candidate remains unaccepted.\n"
            "- Recovering architecture, record-derived targets, and saved edit objects before a smaller reversible adjustment.\n")
```
