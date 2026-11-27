Dose information enters a shared recurrent state, so changing the whole transition could damage many predictions. I’ll first test whether a small value-head edit can use dose information already present in that state. This is promising only if dose-related features explain factual prediction errors on held-out stays.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

print("\n=== REMAINING HEAD/PENDING/LATENT TENSORS ===")
for name in NAMES:
    if name.startswith(("last_", "query_", "type_", "z_", "pending_", "filter_", "monitor_", "time_")):
        print(name, tuple(SHAPES[name]))

print("\n=== VOCABULARY STRUCTURE ===")
print("keys:", list(VOCAB))
for key in ("config", "codes", "normalisation", "normalization", "norm", "norms"):
    if key in VOCAB:
        print(key, VOCAB[key])

print("\n=== FORWARD FILES ===")
for d in sorted(Path("/work").glob("fwd_*")):
    print(d.name, [(p.name, p.stat().st_size) for p in sorted(d.iterdir())])

print("\n=== RETAINED NAMES ===")
print(" ".join(sorted(
    name for name in list(globals())
    if not name.startswith("_") and name not in {"NAMES", "SHAPES", "VOCAB"}
)))

print("\n=== HEAD OBJECTS ===")
for name in ("head_rows", "state_x", "head_weights", "panel", "HC"):
    obj = globals().get(name)
    print(name, type(obj).__name__, getattr(obj, "shape", None))
    if isinstance(obj, pd.DataFrame):
        print("columns:", list(obj.columns))
        print(obj.head(2).to_string())
    elif isinstance(obj, dict):
        print("keys:", list(obj)[:35])
    elif isinstance(obj, np.ndarray):
        print("sample:", obj.reshape(-1)[:8])

for name in ("head_exact", "reconstruct_head"):
    fn = globals().get(name)
    if callable(fn):
        print("\nfunction", name, "args:", fn.__code__.co_varnames[:fn.__code__.co_argcount],
              "referenced names:", fn.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 25; forward 4\n"
        "- Architecture places dose information in a shared recurrent state.\n"
        "- Prefer testing a small value-head correction before risking shared transition weights.\n"
        "- Next test: whether state-derived dose innovations explain original factual prediction errors.\n"
    )
```