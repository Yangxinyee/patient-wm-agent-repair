The recovered notes support an immediate insulin effect near −2 glucose units per dose unit and a norepinephrine effect near +3 MAP units. The conflicting simple state-space fits were rejected, so I won’t use their targets. The output was truncated before the controller details; I’ll request a compact inventory to locate the validated kernels and edit components.

```python
import inspect
import numpy as np

print("=== FUNCTION INVENTORY ===")
fn147 = [
    (n, o) for n, o in list(globals().items())
    if inspect.isfunction(o) and not n.startswith("_")
]
print(", ".join(sorted(n for n, o in fn147))[:2600])
for n, o in sorted(fn147):
    if any(s in n.lower() for s in ("core", "step", "embed", "value", "prelude", "capture", "controller")):
        try:
            print(n, str(inspect.signature(o))[:200])
        except Exception:
            pass

print("\n=== CONTROLLER COMPONENTS ===")
for n in ("P125", "T125", "QE125", "SP125", "Vnew125", "Wrepair125",
          "chosen125", "Ccontroller145", "routes145", "captured_pre118"):
    o = globals().get(n)
    if isinstance(o, np.ndarray):
        print(n, o.shape, "norm", round(float(np.linalg.norm(o)), 5))
    elif isinstance(o, dict):
        print(n, {str(k): (v.shape if hasattr(v, "shape") else type(v).__name__)
                  for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        print(n, type(o).__name__, "length", len(o))
    else:
        print(n, repr(o)[:300])

print("\n=== RELATED VARIABLE NAMES ===")
related147 = [
    n for n in list(globals())
    if n.endswith(("125", "145"))
    or any(s in n.lower() for s in ("kernel", "causal", "target", "effect"))
]
print(", ".join(sorted(related147))[:2300])

print("\n=== COMPACT CAUSAL / ROLLOUT NOTES ===")
notes147 = open("/work/notes.md").read()
selected147 = [
    line for line in notes147.splitlines()
    if any(s in line.lower() for s in
           ("kernel", "accepted", "rollout target", "dose-response", "distributed lag"))
    and not line.startswith("{")
]
print("\n".join(line[:650] for line in selected147)[-2600:])
```
