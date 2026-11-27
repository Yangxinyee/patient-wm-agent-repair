The coordinate checks identify the controller: it retains the original output outside four private directions and replaces their rows with the fitted memory decoder. I’ll now verify a clean injection route and recover the value-head directions. I’ll proceed only if local tests show predictable signal transmission with little non-private disturbance.

```python
import numpy as np
import inspect

print("=== CLEAN WRITE IDENTITIES ===")
print("Cperp @ Uexact norm", np.linalg.norm(Cperp145 @ Uexact145))
print("Ccontroller @ Uexact - R norm",
      np.linalg.norm(Ccontroller145 @ Uexact145 - R118))
for key, route in routes145.items():
    print(key, "private error", round(float(np.linalg.norm(P125.T @ route-np.eye(4))), 6),
          "non-private leak", round(float(np.linalg.norm(Cperp145 @ route)), 6))

print("\n=== VALUE-HEAD TENSORS ===")
for n in NAMES:
    if n.startswith("attr_heads.value_bin"):
        print(n, SHAPES[n])

print("\n=== VALUE ROUTE COMPONENTS ===")
for n in ("Rbase118", "Vsignal118", "CinvJ118", "Jhead118", "Jrows118",
          "derivative118", "escore118", "estate118", "rotation118",
          "score118", "r118", "state_rows118", "hrows118"):
    o = globals().get(n)
    if hasattr(o, "shape"):
        a = np.asarray(o)
        print(n, a.shape, "norm", round(float(np.linalg.norm(a)), 5))
        if a.size <= 16:
            print(a)
    elif isinstance(o, dict):
        print(n, {str(k): (v.shape if hasattr(v, "shape") else type(v).__name__)
                  for k, v in o.items()})
    else:
        print(n, repr(o)[:150])

print("\n=== ESSENTIAL HELPER DEFINITIONS ===")
for n, budget in (("bundle_inputs74", 2300), ("head_inputs96", 1200),
                  ("decode67", 2000)):
    o = globals().get(n)
    if inspect.isfunction(o):
        print("\n", n, inspect.signature(o))
        print(skeleton149(o, budget))

print("\n=== ATTENTION / CONTROLLER NOTE ===")
text151 = open("/work/notes.md").read()
for title in ("## Run 118", "## Run 125"):
    pos = text151.find(title)
    if pos >= 0:
        end = text151.find("\n## ", pos+4)
        print(text151[pos:(end if end >= 0 else pos+1400)][:1100])
```
