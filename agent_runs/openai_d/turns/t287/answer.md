The stronger trial preserved insulin’s response and moved norepinephrine x2 from +1.26 to +1.62, but half-dose barely improved. Cancellation helps modestly; it has not repaired MAP effects. I’ll recover the edit’s construction and training-only fit diagnostics to identify which components consume the factual budget. Any further trial must improve rollout effects while remaining below the NLL limit.

```python
from pathlib import Path
import inspect
import numpy as np

r244_current = {n: load(n).clone() for n in edited()}
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Forward 32–33 / Run 244: cancellation trial validated\n"
        "Current four-head-tensor trial e042bb7889: tuning100–199 factual "
        "NLL delta +0.00471. Tuning0–99 dose shifts insulin -13.610/+6.178; "
        "NE +1.615/-0.693. Cancellation provides modest actual-dose improvement "
        "but NE remains well below provisional targets. Saved exact current "
        "tensors in r244_current; original safe fallback remains available.\n"
    )

print("=== DATA README ===")
print(Path("/data/README.md").read_text()[:2200])
print("\n=== EARLY NOTEBOOK: IDENTIFICATION ===")
print(Path("/work/notes.md").read_text()[:4500])

print("\n=== RECENT OBJECT INTERFACES ===")
for name in sorted(list(globals())):
    if name.startswith(("r232", "r235", "r237", "r239", "r240", "r241", "r242")):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "dict:", list(obj.keys())[:22])
        elif callable(obj):
            print(name, "function:", list(inspect.signature(obj).parameters))
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (int, float, str, bool)):
            print(name, repr(obj)[:120])
        else:
            print(name, type(obj).__name__)

print("\n=== OTHER CANDIDATE/EVALUATOR FUNCTIONS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and any(k in name.lower() for k in
                             ("candidate", "eval", "make", "build", "probe")):
        try:
            print(name, list(inspect.signature(obj).parameters))
        except (TypeError, ValueError):
            pass
```
