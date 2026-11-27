The recovered notes support immediate effects near −2 for insulin and +3.3 for norepinephrine per rate unit. Those are not the accumulated rollout effects. The existing correction nearly matches insulin’s rollout target but costs too much factual likelihood when combined with norepinephrine. I’ll recover the cached evaluator and candidate tensors; any revision must pass both factual and rollout checks.

```python
import inspect
import numpy as np
import torch

print("=== RECENT VARIABLE NAMES ===")
print("\n".join(recent219))

print("\n=== EVALUATOR FUNCTIONS ===")
for k in recent219:
    v = globals()[k]
    if callable(v) and hasattr(v, "__code__"):
        print(k, inspect.signature(v))
        print("uses:", v.__code__.co_names)
        print("locals:", v.__code__.co_varnames[:30])

print("\n=== CACHE AND CANDIDATE DICTIONARIES ===")
for k in recent219:
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "keys:", list(v.keys())[:25])
        if k.startswith(("r210", "c210", "r203", "c203")):
            print("shapes:", {
                str(a): (tuple(b.shape) if hasattr(b, "shape") else type(b).__name__)
                for a, b in list(v.items())[:25]
            })

print("\n=== FOUR EDITED TENSORS ===")
for name in edited():
    t = torch.as_tensor(load(name))
    o = torch.as_tensor(load_original(name))
    d = t - o
    print(name, tuple(t.shape), "delta norm", float(d.norm()))
    if t.ndim == 2:
        print("changed rows:",
              torch.where(torch.any(d != 0, dim=1))[0].tolist()[:40])

print("\n=== OTHER RELEVANT FUNCTIONS / BACKUPS ===")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in ("innovation", "decode", "gate", "candidate", "cache")):
        v = globals()[k]
        if callable(v) and hasattr(v, "__code__"):
            print(k, inspect.signature(v))
        elif isinstance(v, dict) and len(v) <= 15:
            print(k, "dict keys", list(v.keys()))
```
