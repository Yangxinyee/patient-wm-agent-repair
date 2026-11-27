The compensated attention relay has stayed within the factual-loss limit on the tested batch. The saved dose carrier also responds to dose changes, but we have not shown that the relay transmits it. I’ll recover the exact relay and carrier definitions before connecting them; a useful connection must change rollouts without breaking factual fit.

```python
from pathlib import Path
import numpy as np, torch, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== ACTIVE EDITS ===")
for n in edited():
    print(n, SHAPES[n])

print("\n=== RELAY / CARRIER CACHE INVENTORY ===")
_g209 = list(globals().items())
for n, v in sorted(_g209):
    if n.startswith("_"):
        continue
    if any(s in n.lower() for s in ("relay", "uniform", "carrier", "attn")):
        if isinstance(v, np.ndarray) or torch.is_tensor(v):
            print(n, tuple(v.shape), str(v.dtype))
        elif isinstance(v, dict):
            print(n, "dict", {str(k): tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
                             for k, x in list(v.items())[:16]})
        elif callable(v):
            print(n, "function", getattr(getattr(v, "__code__", None), "co_names", ()))
        elif isinstance(v, (int, float, str, list, tuple)):
            print(n, repr(v)[:300])

print("\n=== SAVED CARRIER INSTALLER ===")
if "rp_install" in globals():
    try:
        print(inspect.getsource(rp_install))
    except (OSError, TypeError):
        import dis
        dis.dis(rp_install)

print("\n=== CARRIER BRANCH SPECIFICATIONS ===")
for drug, spec in rp_branch_specs.items():
    print(drug)
    for k, v in spec.items():
        if hasattr(v, "shape"):
            a = np.asarray(v)
            print(" ", k, a.shape,
                  "range", (float(a.min()), float(a.max())) if a.size else None)
        else:
            print(" ", k, repr(v)[:400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 209\n"
            "Relay baseline passed the tested factual bound (+0.00054770 nats/reading). "
            "Recovering exact attention and carrier definitions before connecting them; "
            "no additional weights changed. Counts: 209 Python runs, 19 forwards.\n")
```