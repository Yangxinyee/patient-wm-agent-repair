The architecture confirms that a posterior dose signal can enter the recurrent state through the reserved attention head. That head has passed a factual-fit check, but its effect on rollouts is still untested. I’ll inspect the candidate’s dose decoder and state helpers next; accurate dose contrasts and a verified state update would support calibration, while either mismatch would argue against installing it.

```python
import inspect, re
import numpy as np
import torch

snap158 = dict(globals())

def brief158(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        return {"shape": a.shape, "norm": float(np.linalg.norm(a)),
                "min": float(np.min(a)), "max": float(np.max(a))}
    if isinstance(x, dict):
        return {str(k): (tuple(v.shape) if hasattr(v, "shape")
                        else type(v).__name__) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x),
                "items": [tuple(v.shape) if hasattr(v, "shape")
                          else type(v).__name__ for v in x[:8]]}
    return repr(x)[:120]

print("=== RECENT VARIABLES ===")
for n in sorted(snap158):
    if re.search(r"(148|149|150|151|152|153|154|155|156)$", n):
        print(n, brief158(snap158[n]))

print("\n=== FUNCTION INVENTORY ===")
for n, x in sorted(snap158.items()):
    if inspect.isfunction(x) and not n.startswith("_"):
        try:
            print(n, str(inspect.signature(x)))
        except Exception:
            print(n)

print("\n=== POSTERIOR AND RECURRENT TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "core.blocks", "core.coda",
                            "core.c_out", "core.b_in", "core.delta",
                            "core.a_in", "core.age")):
        print(n, SHAPES[n])

print("\n=== AVAILABLE FORWARD FILES ===")
from pathlib import Path
for k in (1, 2, 7, 10):
    p = Path(f"/work/fwd_{k}")
    if p.exists():
        print(k, [q.name for q in p.iterdir()])

print("\n=== HELPER SOURCE AVAILABILITY ===")
for n in ("posterior_mean", "prelude_parts145", "means105"):
    if n in snap158:
        try:
            s = inspect.getsource(snap158[n])
            print(n, s[:1600])
        except Exception as ex:
            print(n, type(ex).__name__)
```
