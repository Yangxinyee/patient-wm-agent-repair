The fitted kernels suggest a delayed insulin response and a shorter norepinephrine response, but the long-horizon targets remain provisional. No weights have changed. I’ll recover the architecture and retained diagnostics so we can validate those targets and choose a narrowly targeted repair; agreement on held-out records would support proceeding.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED OBJECTS ===")
selected90 = [
    "moment_data", "_moment_inputs", "persistence_repair", "persistence_loss",
    "horizon", "transition73", "pre_context", "head_features_fixed",
    "one_step74", "pair80", "phi74", "prob74", "centres74"
]
for name in selected90:
    obj = globals().get(name)
    if callable(obj):
        print(name, "function",
              "args:", inspect.signature(obj),
              "globals:", obj.__code__.co_names,
              "closure:", obj.__code__.co_freevars)
    elif isinstance(obj, dict):
        print(name, "dict", [
            (str(k), type(v).__name__, getattr(v, "shape", None))
            for k, v in list(obj.items())[:16]
        ])
    else:
        print(name, type(obj).__name__, getattr(obj, "shape", None))

print("\n=== CAUSAL / GATE / POSTERIOR OBJECT NAMES ===")
print(sorted(
    n for n in globals()
    if not n.startswith("_")
    and any(s in n.lower() for s in
            ["gate", "innov", "panel", "kernel", "post", "fixture", "longitud", "ridge87"])
))

print("\n=== WEIGHT NAMES AND SHAPES ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ["action", "transition", "recur", "memory", "post", "prior", "head", "readout"]):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 90\n"
            "Two-pole kernel targets remain provisional, not established truth. "
            "Recovering architecture and retained causal/repair fixtures before "
            "validating sustained effects and choosing a targeted edit. No edits.\n")
```
