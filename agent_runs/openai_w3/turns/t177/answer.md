The retained candidate has two dose-innovation carriers, but the output was cut off before their implementation and the state helpers appeared. No weights changed. I’ll request only those definitions now so we can test the candidate directly; correct dose contrasts would support it, but factual posterior drift would require revision.

```python
import inspect
import numpy as np
import torch

print("=== CANDIDATE ===")
for n in ("post154", "zin154", "zb154", "policy154", "slots154",
          "carrier154", "scale154", "scenarios153", "scenario_names153",
          "reference", "references"):
    if n not in globals():
        continue
    x = globals()[n]
    if isinstance(x, dict):
        print(n, {str(k): (tuple(v.shape) if hasattr(v, "shape")
                          else repr(v)[:70]) for k, v in x.items()})
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        print(n, tuple(x.shape))
    elif isinstance(x, (list, tuple)):
        print(n, "length", len(x), [
            [tuple(y.shape) if hasattr(y, "shape") else repr(y)[:45]
             for y in v] if isinstance(v, (list, tuple)) else repr(v)[:80]
            for v in x[:6]])
    else:
        print(n, repr(x)[:300])

func159 = {n: x for n, x in list(globals().items())
           if inspect.isfunction(x) and not n.startswith("_")}
print("\n=== FUNCTION NAMES ===")
print(", ".join(sorted(func159)))

print("\n=== SELECTED FUNCTION DETAILS ===")
for n, x in sorted(func159.items()):
    if any(s in n.lower() for s in
           ("posterior", "prelude", "state_step", "core_step", "after_e",
            "recurrent", "means105", "readout", "action_embed")):
        print(n, inspect.signature(x), "dependencies:", x.__code__.co_names)

print("\n=== CORE TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "core.blocks",
                            "core.coda", "core.c_out", "core.b_in",
                            "core.delta", "core.a_in", "core.age")):
        print(n, SHAPES[n])

print("\n=== FORWARD 1 FILES ===")
from pathlib import Path
print([p.name for p in Path("/work/fwd_1").iterdir()])
```
