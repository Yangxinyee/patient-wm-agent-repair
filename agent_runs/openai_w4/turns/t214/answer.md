The index check rules out misaligned doses. The newer gated decoder already attenuates dose changes before they reach the model, so I’ll inspect that wiring before adjusting response strength. A useful fix must preserve proportional dose transmission without firing on non-drug bundles.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

items189 = list(globals().items())

def summary189(v):
    if torch.is_tensor(v):
        return f"Tensor {tuple(v.shape)}"
    if isinstance(v, np.ndarray):
        return f"array {v.shape}"
    if isinstance(v, dict):
        return "dict keys=" + repr(list(v.keys())[:18])
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} len={len(v)}"
    if callable(v) and hasattr(v, "__code__"):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "(?)"
        return f"function {sig}; dependencies={v.__code__.co_names}"
    return repr(v)[:180]

print("\nCHANNEL/GATE OBJECTS")
for name, value in items189:
    if name.endswith(("180", "185", "186", "187")):
        print(name, summary189(value))

print("\nRETAINED RAW NE DECODERS")
for name in ("raw_candidates87", "gate_candidates87", "gate_fits180",
             "indices102", "ix180", "channel_weights160", "direction161"):
    if name in globals():
        value = globals()[name]
        print(name, summary189(value))
        if isinstance(value, dict):
            for key, entry in list(value.items())[:12]:
                print(" ", repr(key), summary189(entry))

print("\nCHANNEL CONSTRUCTION SOURCE, IF AVAILABLE")
for name, value in items189:
    if (name.endswith(("186", "187"))
        and callable(value) and hasattr(value, "__code__")):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            continue
        print(name)
        print(source[:9000])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 189\n"
        "Run-188 alignment checks ruled out indexing/rate mismatch. "
        "The run-180 gated decoder itself attenuates imposed NE dose changes, "
        "whereas the older decoder is nearly proportional on its smaller active subset. "
        "Recovering exact channel/gate construction before modifying it; "
        "classification accuracy alone does not establish faithful dose transmission. "
        "No saved weights changed.\n"
    )
```