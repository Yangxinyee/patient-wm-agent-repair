The direct head correction failed the trade-off: candidates large enough to matter damaged factual fit, while the acceptable candidate captured only about 2% of the requested response. I’ll reject that approach. Next I’ll examine whether the existing dose-residual branch can feed a dedicated state carrier. This is worth testing only if its factual signal is small and its dose-change signal generalizes to held-out stays.

```python
import numpy as np
import dis
from pathlib import Path

def brief194(v):
    if hasattr(v, "shape"):
        a = np.asarray(v)
        return {"shape": a.shape, "rms": float(np.sqrt(np.mean(a*a))),
                "min": float(a.min()), "max": float(a.max())}
    if isinstance(v, dict):
        return {k: brief194(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [brief194(x) for x in v]
    if isinstance(v, (float, int, str, bool, np.integer, np.floating)):
        return v
    return type(v).__name__

print("BRANCH SPECS")
print(brief194(rp_branch_specs))
print("\nGATED FITS")
print(brief194(rp_gate_fits))
print("\nRESERVED HIDDEN UNITS", rp_reserved)

print("\nINSTALLER BYTECODE")
inst = list(dis.Bytecode(rp_install))
for x in inst[:145]:
    print(x.opname, str(x.argrepr)[:120])

print("\nOBSERVATION-PATH CACHE NAMES")
found = []
for k, v in list(globals().items()):
    if any(s in k.lower() for s in ("direction", "eplus", "eminus", "de_gl", "de_map", "e_gl", "e_map", "jacob")):
        found.append((k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__))
print(found[:35])
print("\nPRELUDE TENSOR NAMES")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("core.prelude")])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 194\nDirect output-head contrast fit rejected: factual-safe lambda=30 captured only ~2% of contrast targets; stronger fits violated the NLL bound. Recovering gated dose residuals for a separate latent carrier and direct state transmission. No edits installed.\n")
```
