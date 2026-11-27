The strong edit produces alternating hourly responses and a large drift in its recorded-dose rollout. That supports a feedback problem, not just insufficient strength. I’ll prepare the smaller code- and register-protected candidate, whose CPU NLL cost is within the limit. A GPU test must show a stronger, stable insulin response; another oscillation or an NLL failure would reject it.

```python
import inspect
import numpy as np
import torch
from pathlib import Path

print("=== SYNTHETIC-CONTRAST CONSTRUCTION ===")
for key in ["sc_names", "sc_nodes", "sc_old_scale", "sc_old_b",
            "sc_gate_b", "sc_bias_comp", "sc_rank"]:
    if key in globals():
        v = globals()[key]
        if hasattr(v, "shape"):
            a = np.asarray(v)
            print(key, "shape", a.shape, "values", a if a.size <= 20 else
                  {"norm": float(np.linalg.norm(a)), "min": float(a.min()), "max": float(a.max())})
        else:
            print(key, repr(v)[:500])

print("sc_evaluate signature:", inspect.signature(sc_evaluate))
print("references:", sc_evaluate.__code__.co_names)
print("locals:", sc_evaluate.__code__.co_varnames)
print("constants:", repr(sc_evaluate.__code__.co_consts)[:1400])

trial_r3 = r3_candidates[(0.1, 1.0)]
print("=== SELECTED SMALL CANDIDATE ===")
for key, v in trial_r3.items():
    if hasattr(v, "shape"):
        a = np.asarray(v)
        print(key, "shape", a.shape, "norm", float(np.linalg.norm(a)))
    else:
        print(key, repr(v)[:600])

print("=== CACHE SHAPES ===")
for key in ["sc_fact", "sc_syn", "sc_original", "sc_checkpoint"]:
    v = globals()[key]
    print(key, "keys", list(v.keys())[:12])
    for kk, vv in list(v.items())[:3]:
        if isinstance(vv, dict):
            print(" ", kk, {k: getattr(a, "shape", type(a).__name__) for k, a in vv.items()})
        else:
            print(" ", kk, getattr(vv, "shape", type(vv).__name__))

print("=== CURRENT CHECKPOINT PRESERVED ===")
restart_checkpoint = {name: load(name).clone() for name in edited()}
print({name: tuple(t.shape) for name, t in restart_checkpoint.items()})

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Rollout feedback audit; Python 177, forwards 11/50\n"
            "- Record-derived insulin RD estimate: beta=-1.9287, persistence=0.83184; doubling target about -11, half about +5.5.\n"
            "- beta=-24 GPU response oscillates by hour; its generated recorded-dose glucose shifts by -29.38 on average.\n"
            "- beta=-2 response is stable but weak; recorded-dose generated glucose shifts by -6.00.\n"
            "- Preparing r3 lambda=0.1, scale=1 candidate for paired GPU validation; synthetic probes are design evidence only.\n")
```
