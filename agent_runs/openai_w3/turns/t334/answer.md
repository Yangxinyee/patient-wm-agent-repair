```python
import os, re, inspect, numpy as np
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text()[:18000])

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION ===")
print(notes_text[:11000])
print("\n=== PRIOR VALIDATION / ROLLOUT SUMMARIES ===")
hits = [
    line for line in notes_text.splitlines()
    if re.search(r"rollout|mean_shift|nats|readouts|reference|carrier|checkpoint", line, re.I)
]
for line in hits[:-35]:
    print(line[:1000])

print("\n=== RETAINED FUNCTIONS ===")
fnames = sorted(
    k for k, v in list(globals().items())
    if inspect.isfunction(v) and not k.startswith("_")
)
print(fnames)
print("\n=== RELEVANT VARIABLES ===")
for k, v in sorted(list(globals().items())):
    if re.search(r"checkpoint|carrier|candidate|direction|dose|gamma|target|coef|beta|ids|replay", k, re.I):
        if isinstance(v, dict):
            print(k, "dict", len(v), "keys", list(v)[:18])
        elif isinstance(v, np.ndarray):
            print(k, "array", v.shape)
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:300])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v), repr(v[:8])[:350])
        elif inspect.isfunction(v):
            print(k, "function", inspect.signature(v))
        else:
            print(k, type(v).__name__, str(getattr(v, "shape", "")))

print("\n=== FACTUAL PASS 29 ===")
p29 = Path("/work/fwd_29")
for p in sorted(p29.glob("*")):
    print(p.name)
    if p.suffix == ".json":
        print(p.read_text()[:6000])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 290; forwards 30. Alpha=.005 rollout (-8.092,+1.383,+.104,-1.119) confirms contraction alone cannot repair all dose directions. Recovering earlier validated candidates and exact editing helpers before selecting a drug-specific reversible test. No weights changed.\n")
```
