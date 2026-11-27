The latest test improved norepinephrine sensitivity but barely changed insulin sensitivity. Those local diagnostics do not establish that the saved edit is safe or effective. I’ll recover the exact computation path and retained evidence before choosing the next edit; a useful candidate must respond correctly to dose changes and preserve factual predictions.

```python
from pathlib import Path
import inspect, re
import numpy as np
import torch

arch189 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch189[:18000])
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text()[:7000])

print("\n=== RELEVANT RETAINED OBJECTS ===")
keys189 = [k for k in list(globals())
           if not k.startswith("_") and re.search(
               r"(187|188|139|candidate|token|route|causal|effect|persist|local.*head|head.*local|replay|scratch|dose)",
               k, re.I)]
for k in keys189[-100:]:
    v = globals()[k]
    if inspect.isfunction(v):
        sig = inspect.signature(v)
        args = [p.name for p in sig.parameters.values()]
        print(k, "function", args)
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v)[:16])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif isinstance(v, (str, int, float, bool, np.number)):
        print(k, repr(v)[:180])
    else:
        print(k, type(v).__name__)

print("\n=== SAVED EDITS ===")
print(edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n\n## Run 189: restart and bounded recovery\n"
            "- Run 188 improved local norepinephrine routing but not insulin; local KL is not an official factual-fit result.\n"
            "- Recovering exact architecture, record schema, and retained routing interfaces before selecting another edit.\n"
            "- No weights changed. Python runs: 189; forward passes: 10.\n")
```