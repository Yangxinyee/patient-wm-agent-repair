The scaling audit corrected the diagnostics, but it did not establish a better edit: the innovation fits still have factual leakage and asymmetric dose responses. The current trial remains unaccepted. I’ll recover the architecture and retained fitting objects, then seek a correction that improves both insulin readouts without exceeding the factual-NLL tolerance.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRETAINED OBJECTS")
items = list(globals().items())
tokens = ("trial", "slot", "causal", "effect", "panel", "dose", "contrast",
          "prelude", "innovation", "state", "sensitivity", "anchor", "direction")
for name, obj in sorted(items):
    if name.startswith("_") or not any(t in name.lower() for t in tokens):
        continue
    if isinstance(obj, np.ndarray):
        print(name, "array", obj.shape, obj.dtype)
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, "columns:", list(obj.columns))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj.keys())[:16])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, "length", len(obj))
    elif isinstance(obj, (float, int, str, np.number)):
        print(name, repr(obj)[:240])
    elif callable(obj):
        try:
            print(name, "function", inspect.signature(obj))
        except Exception:
            print(name, "callable")
    else:
        print(name, type(obj).__name__)

print("\nAVAILABLE HELPERS")
for name, obj in sorted(items):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 53: restart recovery and acceptance criterion\n"
        "Run 52 corrected feature scaling but did not establish a better edit. "
        "The existing core.b_in.weight trial remains unaccepted because its "
        "50-stay factual NLL increased by 0.00597. Recovering architecture and "
        "retained objects before improving selectivity and testing both dose "
        "directions. No weights changed. Python runs: 53; forwards: 5.\n"
    )
```