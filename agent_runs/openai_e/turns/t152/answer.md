The output confirms separate query and memory projections, but it was truncated before showing their shapes. The mismatch may be a missing memory batch dimension. I’ll inspect the retained inputs compactly; that diagnosis would be confirmed by memory lacking the query’s batch dimension and refuted if the dimensions already agree.

```python
import numpy as np, torch, inspect
from pathlib import Path

print("Architecture input and state:")
print("\n".join(lines130[7:54]))
print("\nDose-edit semantics:")
print("\n".join(lines130[79:]))

snapshot131 = dict(globals())
print("\nRelevant function inventory:")
for n131, v131 in snapshot131.items():
    if callable(v131) and hasattr(v131, "__code__") and (
        n131.startswith(("core_", "attn", "layer", "recon", "bundle", "read"))
        or n131.endswith(("127", "128", "129"))
    ):
        print(n131, v131.__code__.co_varnames[:v131.__code__.co_argcount])

print("\nRecent arrays:")
count131 = 0
for n131, v131 in snapshot131.items():
    if isinstance(v131, (np.ndarray, torch.Tensor)) and (
        n131.endswith(("121", "125", "126", "127", "128", "129", "95"))
        or any(s in n131.lower() for s in ["gpu", "hprev", "hnew", "prelude"])
    ):
        print(n131, tuple(v131.shape), str(v131.dtype))
        count131 += 1
        if count131 >= 65:
            break

print("\nRecent containers:")
for n131, v131 in snapshot131.items():
    if n131.endswith(("127", "128", "129")) and isinstance(v131, (dict, list, tuple)):
        if isinstance(v131, dict):
            print(n131, "dict", len(v131), "keys", list(v131)[:12])
        else:
            print(n131, type(v131).__name__, len(v131))

print("\nCore layer code metadata:")
for n131 in ["core_layer", "core_lin", "core_attn", "core_step"]:
    v131 = snapshot131.get(n131)
    if callable(v131) and hasattr(v131, "__code__"):
        c131 = v131.__code__
        print(n131, "locals", c131.co_varnames, "names", c131.co_names,
              "constants", tuple(x for x in c131.co_consts
                                 if isinstance(x, (str, int, float, tuple))))

notes131 = Path("/work/notes.md").read_text()
print("\nEarlier calibration notes:")
selected131 = [p for p in notes131.split("\n## ")
               if any(s in p.lower() for s in ["baseline77", "insulin candidate",
                                              "target", "causal effect"])
               and not p.startswith(("Run 1", "Runs 1"))]
print("\n\n".join(selected131[-5:])[:2400])
with open("/work/notes.md", "a") as f131:
    f131.write("\n## Run 131: compact attention/input inventory\n"
               "Run130 output truncated during bytecode; no numerical test or edit. "
               "Recovering compact input shapes, core helper metadata and dose-edit "
               "semantics to fix the reconstruction. Python runs: 131; forwards: 15.\n")
```