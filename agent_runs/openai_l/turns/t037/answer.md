The best reconstruction now disagrees with the forward tool by roughly the amount expected from latent sampling, so it is plausible enough to stop the broad search. I’ll recover the exact decoder and action pathways before designing an edit. A useful edit must change dose sensitivity while leaving predictions on recorded doses nearly unchanged.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED BEST DECODER ===")
print("r32_best:", repr(globals().get("r32_best")))

print("\n=== FUNCTIONS AVAILABLE ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, sig)

print("\n=== RETAINED DATA / STATE OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or inspect.isfunction(v):
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict) and (
        any(s in k.lower() for s in ("state", "head", "effect", "kernel", "spec", "config"))
        or k in ("r32_details", "r32_best")
    ):
        print(k, "dict keys:", list(v.keys())[:25])

print("\n=== ACTION / STATE / DECODER TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in (
        "action", "posterior", "prior", "recur", "gru", "cell",
        "last_numeric", "value_head", "readout", "out_proj"
    )):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 33 / restart recovery\n"
            "Best CPU decoder MC disagreement is approximately sampling noise; "
            "recovering exact architecture, retained functions and action/state tensors "
            "to design a factual-preserving causal correction. No edits yet.\n")
```