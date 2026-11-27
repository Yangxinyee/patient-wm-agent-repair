The recovered arrays provide a way to test the proposed state wire locally before spending a forward pass. I’ll check which matrices decode memory, which write it, and how they relate to the existing weights. A successful wire should reproduce the intended memory amplitudes with little leakage; failing that test would rule out this construction.

```python
import numpy as np
import inspect

print("=== WEIGHT AND COMPONENT NORMS ===")
for n in ("core.b_in.weight", "core.c_out.weight", "core.delta_in",
          "core.prelude_norm.weight"):
    a = np.asarray(load_original(n))
    print(n, a.shape, "norm", round(float(np.linalg.norm(a)), 5),
          "range", (round(float(a.min()), 4), round(float(a.max()), 4)))
for n in ("bn125", "constant125", "constvec125", "constraints125",
          "P125", "T125", "QE125", "SP125", "Ccontroller145", "Cperp145"):
    a = globals().get(n)
    if hasattr(a, "shape"):
        a = np.asarray(a)
        print(n, a.shape, "norm", round(float(np.linalg.norm(a)), 5))

print("\n=== COORDINATE IDENTITIES ===")
I150 = np.eye(4)
for n, a in (("T", T125), ("Uexact", Uexact145), ("R", R118),
             ("QE", QE125), ("SP", SP125)):
    for label, b in (("P.T", P125.T), ("QE.T", QE125.T), ("R.T", R118.T)):
        m = b @ a
        print(label + "@" + n, np.round(m, 3))
print("decoder regression RMS",
      np.sqrt(np.mean((Xstate125 @ P125 - Ystate125)**2, axis=0)))

print("\n=== CONTROLLER COMPARISONS ===")
for n, a in (
    ("b_in", np.asarray(load_original("core.b_in.weight"))),
    ("c_out", np.asarray(load_original("core.c_out.weight"))),
    ("T@P.T", T125 @ P125.T),
    ("R@P.T", R118 @ P125.T),
    ("Uexact@P.T", Uexact145 @ P125.T),
    ("Cperp+T@P.T", Cperp145 + T125 @ P125.T),
    ("Cperp+R@P.T", Cperp145 + R118 @ P125.T),
    ("Cperp+Uexact@P.T", Cperp145 + Uexact145 @ P125.T),
):
    print(n, "difference", round(float(np.linalg.norm(Ccontroller145-a)), 6))

print("\n=== BATCH PREPARATION HELPERS ===")
for n in ("prepare103", "bundle_inputs73", "bundle_inputs74", "unpack49",
          "head_inputs96", "evaluate83"):
    o = globals().get(n)
    if inspect.isfunction(o):
        print(n, inspect.signature(o))
        if n == "prepare103":
            print(skeleton149(o, 1800))

print("\n=== SELECTED CACHE NAMES ===")
print(", ".join(sorted(n for n in list(globals())
                      if n.endswith(("118", "125", "145"))
                      and ("J" in n or "C" in n or "bn" in n or "controller" in n))))
```
