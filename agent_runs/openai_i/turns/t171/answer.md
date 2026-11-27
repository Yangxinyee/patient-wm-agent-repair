The last probe failed because the readout expected 17 inputs but received 16; it produced no response estimate. I’ll inspect the input layout and retained helpers, then correct and validate the readout. Agreement with exposed model predictions would support using it to choose an edit; disagreement would rule that out.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

arch154 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch154)

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("head", "readout", "z_post", "z_prior", "gru", "lstm", "cell", "transition")):
        print(name, SHAPES[name])

print("\n=== RETAINED HELPERS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and any(s in name.lower() for s in
                                      ("153", "readout", "recur", "step", "context", "encode")):
        print(name, inspect.signature(obj))
        print("locals:", obj.__code__.co_varnames)
        print("constants:", repr(obj.__code__.co_consts)[:1400])

print("\n=== RETAINED ARRAY INVENTORY ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, dict) and any(s in name.lower() for s in
                                     ("153", "state", "cache", "feature", "response", "weight")):
        print(name, "dict keys:", list(obj)[:30])

print("\nACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 154 — readout dimension recovery\n"
            "- Python 153 failed before producing response estimates: 16 inputs versus 17 required.\n"
            "- Rereading the architecture and retained helper contracts before fixing the input layout.\n"
            "- No weights changed; finite-difference directions remain unvalidated.\n")
```