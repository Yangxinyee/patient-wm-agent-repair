The recovered records show why raw dose–outcome associations were misleading: adjustment reversed insulin’s apparent direction, and norepinephrine required richer history controls. I’ll inspect the existing head edit and cancellation fit before changing weights, then screen amplitude trade-offs. A useful candidate would retain near-target insulin effects while freeing enough factual-fit budget for stronger MAP responses.

```python
import numpy as np
import torch
import dis
from pathlib import Path

def r245_summary(obj):
    if isinstance(obj, dict):
        return {str(k): (tuple(v.shape) if hasattr(v, "shape")
                         else f"{type(v).__name__}:{len(v)}" if isinstance(v, (list, dict))
                         else repr(v)[:90])
                for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [r245_summary(x) if isinstance(x, dict)
                else (type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else str(x)[:90])
                for x in obj]
    return (type(obj).__name__, tuple(obj.shape) if hasattr(obj, "shape") else str(obj)[:180])

for name in ["r239_rows", "r240_ne_rows", "r240_keep", "r240_q",
             "r240_regressions", "r242_verified", "c210_train", "c210_tune",
             "cache231", "r240_pre", "r240_baselines"]:
    obj = globals().get(name)
    if name == "r240_regressions":
        print(name, "selected:", r245_summary(obj[(16, .001)]))
    elif name == "r240_pre":
        print(name, {k: r245_summary(v) for k, v in obj.items()})
    elif name == "r240_q":
        print(name, np.asarray(obj))
    elif name == "r240_keep":
        print(name, "excluded:", np.setdiff1d(np.arange(192), obj))
    else:
        print(name, r245_summary(obj))

print("\n=== HELPER BYTECODE ===")
for name in ["r240_loss", "r240_prepare", "c210_eval"]:
    fn = globals()[name]
    print("\n", name, "variables:", fn.__code__.co_varnames,
          "globals:", fn.__code__.co_names)
    print(dis.Bytecode(fn).dis()[:2800])

print("\n=== CURRENT OUTPUT-ROW GEOMETRY ===")
W = load("attr_heads.value_bin.2.weight").numpy()
Wsafe = r235_safe["attr_heads.value_bin.2.weight"].numpy()
Worig = load_original("attr_heads.value_bin.2.weight").numpy()
for j in r239_rows:
    print(int(j), "current:", np.round(W[:, j], 3),
          "safe-relative norm:", float(np.linalg.norm(W[:, j] - Wsafe[:, j])),
          "original norm:", float(np.linalg.norm(Worig[:, j])))

print("\n=== LONG-HORIZON IDENTIFICATION NOTES ===")
notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes)
picked = [s for s in sections
          if any(w in s.lower() for w in ("impulse response", "distributed lag",
                                         "long-horizon identification", "persistence"))]
for s in picked[:4]:
    print(s[:1100])
```
