The kernels show a delayed, multi-hour insulin response and a faster-decaying norepinephrine response; a simple one-step correction would miss that distinction. The controller has four channels, which may be enough to approximate both. I’ll recover the state and readout interfaces next, then test whether those channels can carry the kernels while remaining quiet on factual records.

```python
import dis, inspect
import numpy as np

def skeleton149(func, limit=1800):
    groups, current = [], []
    line = None
    for ins in dis.get_instructions(func):
        if ins.starts_line is not None:
            if current:
                groups.append(f"{line}: " + " ".join(current))
            line, current = ins.starts_line, []
        if ins.opname not in ("RESUME", "CACHE", "NOP", "EXTENDED_ARG"):
            current.append(ins.opname + (":" + ins.argrepr if ins.argrepr else ""))
    if current:
        groups.append(f"{line}: " + " ".join(current))
    return "\n".join(groups)[:limit]

print("=== COMPACT HELPER DEFINITIONS ===")
for n, limit in (("w63", 550), ("linear63", 650), ("context64", 1100),
                 ("transition64", 2800), ("decode67", 850)):
    o = globals().get(n)
    if inspect.isfunction(o):
        print("\n", n, inspect.signature(o))
        print(skeleton149(o, limit))

print("\n=== READOUT COMPONENTS ===")
for n in ("Ustate118", "Uscore118", "R118", "Chead118", "Jselected118",
          "responses118", "route122", "transmit122", "d122"):
    o = globals().get(n)
    if hasattr(o, "shape"):
        a = np.asarray(o)
        print(n, a.shape, "norm", round(float(np.linalg.norm(a)), 4))
        if a.size <= 20:
            print(a)
    elif isinstance(o, dict):
        print(n, {str(k): (v.shape if hasattr(v, "shape") else repr(v)[:80])
                  for k, v in o.items()})
    else:
        print(n, repr(o)[:180])

print("\n=== STATE / EVALUATION CACHE INVENTORY ===")
for n, o in sorted(list(globals().items())):
    if n.endswith(("64", "65", "73", "74", "81", "83", "96", "102", "103", "116")):
        if inspect.isfunction(o):
            continue
        if hasattr(o, "shape"):
            print(n, type(o).__name__, tuple(o.shape))
        elif isinstance(o, dict):
            print(n, "dict", list(o.keys())[:8])
        elif isinstance(o, (list, tuple)):
            print(n, type(o).__name__, len(o))
```
