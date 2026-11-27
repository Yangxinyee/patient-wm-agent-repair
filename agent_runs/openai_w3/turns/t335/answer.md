The stronger contraction reduced the doubled-insulin response to −8.09, but barely improved half-dose insulin or norepinephrine. This argues against fixing all four responses with one scale factor. I’ll recover the earlier results and editing helpers to look for a drug-specific correction. Improvement in both dose directions, without exceeding the factual-loss limit, would support that approach.

```python
import os, re, inspect, numpy as np
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text()[:13000])

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION ===")
print(notes_text[:5500])
print("\n=== EARLIER VALIDATION / ROLLOUT SUMMARIES ===")
hits = [
    line for line in notes_text.splitlines()
    if re.search(r"rollout.*\(|shifts.*\(|readouts|reference shifts|causal.*coefficient|identified", line, re.I)
]
print("\n".join(line[:650] for line in hits)[:10000])

print("\n=== RETAINED FUNCTIONS ===")
fnames = sorted(
    k for k, v in list(globals().items())
    if inspect.isfunction(v) and not k.startswith("_")
)
print(fnames)
print("\n=== RELEVANT VARIABLES ===")
inventory = []
for k, v in sorted(list(globals().items())):
    if re.search(r"checkpoint|carrier|candidate|direction|dose|gamma|target|coef|beta|ids|replay", k, re.I):
        if isinstance(v, dict):
            inventory.append(f"{k}: dict {len(v)}, keys {list(v)[:12]}")
        elif isinstance(v, np.ndarray):
            inventory.append(f"{k}: array {v.shape}")
        elif isinstance(v, (int, float, str)):
            inventory.append(f"{k}: {repr(v)[:180]}")
        elif isinstance(v, (list, tuple)):
            inventory.append(f"{k}: {type(v).__name__} {len(v)}, {repr(v[:5])[:220]}")
        elif inspect.isfunction(v):
            inventory.append(f"{k}: function {inspect.signature(v)}")
        else:
            inventory.append(f"{k}: {type(v).__name__} {getattr(v, 'shape', '')}")
print("\n".join(inventory)[:9000])

print("\n=== FACTUAL PASS 29 ===")
for p in sorted(Path("/work/fwd_29").glob("*")):
    print(p.name)
    if p.suffix == ".json":
        print(p.read_text()[:4000])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 290; forwards 30. Alpha=.005 rollout (-8.092,+1.383,+.104,-1.119) confirms contraction alone cannot repair all dose directions. Recovering earlier validated candidates and exact editing helpers before selecting a drug-specific reversible test. No weights changed.\n")
```
