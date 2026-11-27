The broadened decoder improves the immediate norepinephrine response, but most tested settings exceed the factual-loss budget. The smallest candidate looks promising offline; it has not been saved or validated by rollout. I’ll recover its fitting objects and the architecture before making another edit. It remains viable only if independent factual tests stay within the limit.

```python
from pathlib import Path
import inspect, re
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== CURRENT EDITS ===")
print(edited())

notes264 = Path("/work/notes.md").read_text()
print("=== NOTEBOOK HEADINGS ===")
print("\n".join(x for x in notes264.splitlines() if x.startswith("#")))
print("=== EARLY NOTEBOOK CONTEXT ===")
print(notes264[:6500])

snapshot264 = dict(globals())
print("=== LIVE OBJECT NAMES ===")
print(", ".join(sorted(
    k for k, v in snapshot264.items()
    if not k.startswith("_") and not inspect.ismodule(v)
)))

print("=== RECENT FITTING OBJECTS ===")
for k, v in sorted(snapshot264.items()):
    if re.search(r"26[0-3]|25[5-9]|grid242|backup248|fitmask", k, re.I):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:35])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "length", len(v), repr(v[:4])[:700])
        elif callable(v):
            print(k, "function", getattr(v, "__code__", None).co_varnames
                  if hasattr(v, "__code__") else "")
        else:
            print(k, type(v).__name__, repr(v)[:500])

print("=== FIT/EVALUATION FUNCTIONS ===")
for k, v in sorted(snapshot264.items()):
    if inspect.isfunction(v) and re.search(
        r"fit|eval|head|candidate|pair|decoder|metric|nll|build|solve|stage", k, re.I
    ):
        print(k, inspect.signature(v))
        try:
            print(inspect.getsource(v)[:11000])
        except (OSError, TypeError):
            print("locals:", v.__code__.co_varnames)
            print("names:", v.__code__.co_names)
            print("constants:", repr(v.__code__.co_consts)[:2200])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 264 / restart recovery\n"
            "Run263 broadened-decoder .3/.3 has offline factual deltas "
            "+.001399/+.004756/+.004258 and stronger NE immediate slopes. "
            "Larger settings exceed the factual budget. No weights changed; "
            "recovering architecture and live fitting objects before validation.\n")
```