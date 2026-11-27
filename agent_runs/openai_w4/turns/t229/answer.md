The NE addition passes factual fit but gives the wrong rollout sign. I’ll inspect the exact channel wiring and response calibration. Reliable dose transmission with reversed downstream responses would implicate the response direction; unreliable transmission would implicate the decoder.

```python
from pathlib import Path
import inspect, re
import numpy as np
import torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

def brief201(x, depth=0):
    if torch.is_tensor(x):
        a = x.detach().cpu().numpy()
        return {"shape": list(a.shape), "norm": float(np.linalg.norm(a))}
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "norm": float(np.linalg.norm(x))} if x.dtype.kind in "fiu" else {"shape": list(x.shape)}
    if isinstance(x, dict):
        if depth >= 2:
            return {"keys": list(x)[:16], "size": len(x)}
        return {str(k): brief201(v, depth+1) for k, v in list(x.items())[:20]}
    if isinstance(x, (list, tuple)):
        return [brief201(v, depth+1) for v in x[:12]]
    if callable(x):
        try:
            return {"signature": str(inspect.signature(x)), "globals": list(x.__code__.co_names)}
        except Exception:
            return str(type(x))
    return x if isinstance(x, (str, int, float, bool, type(None), np.number)) else str(type(x))

g201 = dict(globals())
print("\nRECENT CONSTRUCTORS / EVALUATORS")
for n, x in sorted(g201.items()):
    if inspect.isfunction(x) and (
        re.search(r"(18[0-9]|19[0-9]|200)$", n)
        or any(s in n.lower() for s in ["dual", "prelude", "factual", "predict", "action", "response"])
    ):
        print(n, brief201(x))

print("\nTARGETS AND CURRENT CANDIDATE")
for n in [
    "readout_estimates137", "kernels137", "results194",
    "directions_ne187", "J187", "ins_direction187",
    "backup_passing200", "backup_before195", "backup_dual198",
    "submitted193", "reference_bank125"
]:
    if n in g201:
        print(n, brief201(g201[n]))

print("\nRECENT VARIABLE NAMES")
print(" ".join(sorted(n for n in g201 if re.search(r"(193|194|195|198|199|200)$", n))))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 201\nRecovered architecture, record-based targets and exact dual-channel construction/evaluation interfaces to diagnose NE rollout sign reversal. No weights changed. Passing factual candidate remains backed up in backup_passing200.\n")
```