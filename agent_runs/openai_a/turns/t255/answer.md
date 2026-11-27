Scale 7 improves the insulin shifts to −8.63 and +3.78, while barely changing norepinephrine’s readouts. That is useful progress, but factual-fit safety remains unresolved. I’ll save this trial and recover the causal evidence and editing helpers; broader NLL failure or unsupported effect estimates would rule out accepting it.

```python
import torch
import inspect
from pathlib import Path

scale7_checkpoint = {
    n: (load(n).clone() if torch.is_tensor(load(n)) else load(n).copy())
    for n in edited()
}

with open("/work/notes.md", "a") as f:
    f.write("""
## Scale-7 paired validation; Python 221, forwards 25/50
- Saved scale7_checkpoint; edit hash db4975dc63.
- First-100 tuning factual: original 1.63461, edited 1.63931, delta +0.00470.
- Exact paired rollout: insulin x2 -8.628, half +3.777; NE x2 -0.504, half -0.607.
- Improves insulin over stronger6, but factual margin is thin. NOT accepted pending broader factual validation.
- stronger6_checkpoint remains the protected fallback.
""")

print("HELPERS (parameter names only)")
for _n, _v in list(globals().items()):
    if inspect.isfunction(_v) and getattr(_v, "__module__", "") == "__main__":
        print(_n, "(" + ", ".join(inspect.signature(_v).parameters) + ")")

print("\nRELEVANT VARIABLE NAMES")
print(sorted(
    n for n in globals()
    if not n.startswith("_")
    and any(s in n.lower() for s in (
        "checkpoint", "cache", "ridge", "gate", "contrast",
        "target", "rd", "whiten", "select", "regress", "panel"
    ))
))

print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:6800])
```
