The architecture confirms that dose effects can recur through hidden state and generated readings. The early records support a negative insulin effect, but one-hour estimates do not establish the 13–24-hour target. I’ll retrieve the longitudinal estimates and existing edit functions; those will distinguish an incorrect effect strength from distorted rollout dynamics.

```python
import re, inspect
from pathlib import Path

notes227 = Path("/work/notes.md").read_text()
print("=== LONGITUDINAL EVIDENCE / TARGET NOTES ===")
sections227 = re.split(r"(?=^## )", notes227, flags=re.M)
hits227 = [s for s in sections227
           if re.search(r"target|kernel|distributed.lag|longitudinal|instrument|persistent|cumulative", s, re.I)]
for s227 in hits227:
    if not re.search(r"## Run (20[0-9]|21[0-9]|22[0-9])\b", s227):
        print(s227[:850])
        if sum(len(s[:850]) for s in hits227[:hits227.index(s227)+1]) > 3800:
            break

print("\n=== KERNEL OBJECT NAMES ===")
keys227 = sorted(k for k in list(globals())
                 if not k.startswith("_") and
                 re.search(r"target|kernel|causal|effect|replay|candidate|backup|ridge|channel|impulse|dual|passing|weights|head_eval|r225", k, re.I))
print(", ".join(keys227)[:2500])

print("\n=== CALLABLE INTERFACES ===")
for k227 in keys227:
    v227 = globals()[k227]
    if callable(v227):
        try:
            print(k227, inspect.signature(v227))
        except Exception:
            pass
print("\n=== SMALL TARGET OBJECTS ===")
for k227 in keys227:
    v227 = globals()[k227]
    if re.search(r"target|kernel|causal", k227, re.I) and not callable(v227):
        if isinstance(v227, dict):
            print(k227, "keys:", list(v227)[:20])
        elif hasattr(v227, "shape"):
            print(k227, "shape:", tuple(v227.shape))
            if getattr(v227, "size", 1000) < 60 and not callable(getattr(v227, "size", None)):
                print(v227)
        else:
            print(k227, repr(v227)[:200])

with open("/work/notes.md", "a") as f227:
    f227.write("\n## Run 227\nArchitecture recovery confirms dose effects propagate through recurrent state and generated readings. Retrieved compact longitudinal evidence and edit interfaces; immediate observational estimates alone are not treated as rollout targets. No saved weights changed.\n")
```