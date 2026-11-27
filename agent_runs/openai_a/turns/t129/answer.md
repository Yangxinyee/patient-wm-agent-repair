The records support an immediate insulin effect near −1.93 per unit/hour and persistence near 0.83, using policy-threshold estimates. Norepinephrine remains less securely identified. I’ll inspect the reconstruction helpers and actual bundles now; this should reveal whether the remaining error comes from omitted events or incorrect encoding.

```python
import dis
import numpy as np
import pandas as pd

print("=== RECONSTRUCTION GLOBALS ===")
for n, v in sorted(list(globals().items())):
    if n.startswith("rm_") or n in ("ctx_spec", "attr_spec"):
        if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(n, type(v).__name__, tuple(v.shape))
        elif not callable(v):
            txt = repr(v)
            print(n, txt[:700])
        elif n in ("rm_set", "rm_insulin_atoms"):
            print(n, "names:", v.__code__.co_names, "constants:", repr(v.__code__.co_consts)[:1000])

print("\n=== SET ENCODER IMPLEMENTATION ===")
try:
    print(inspect.getsource(rm_set))
except Exception:
    dis.dis(rm_set)

print("\n=== EMBEDDING TABLES ===")
for n in NAMES:
    if n.startswith("embed.") or n.startswith("core.gap") or n.startswith("core.z_post"):
        print(n, SHAPES[n])
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])
print("numeric_note:", VOCAB["numeric_note"])
for k, v in list(VOCAB["numeric"].items())[:3]:
    print("numeric", k, repr(v)[:1400])

print("\n=== RECORD TABLES IN KERNEL ===")
for n, v in sorted(list(globals().items())):
    if isinstance(v, pd.DataFrame) and (
        len(v) > 10000 or any(s in n.lower() for s in ("measure", "bundle", "recon"))
    ):
        print(n, v.shape, list(v.columns)[:18])
```
