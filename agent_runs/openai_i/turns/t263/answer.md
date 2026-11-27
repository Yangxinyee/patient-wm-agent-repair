Most damage appears before the first positive drug dose, so accumulation alone cannot explain the failure. The repair is injecting a background signal where treatment should contribute nothing. I’ll inspect the private-channel construction and retained bundle inputs; the next design must suppress that signal on non-drug bundles.

```python
print("=== RETAINED FUNCTIONS ===")
print([k for k, v in list(globals().items())
       if callable(v) and hasattr(v, "__code__") and k[-1:].isdigit()])

print("\n=== INPUT / ROUTING OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.endswith(("196", "199", "201", "208", "209", "221", "222")):
        if isinstance(v, np.ndarray):
            print(k, "ARRAY", v.shape, v.dtype)
        elif hasattr(v, "shape") and not callable(v):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "DICT", list(v.keys())[:18])
        elif callable(v) and hasattr(v, "__code__"):
            print(k, "FUNCTION",
                  v.__code__.co_varnames[:v.__code__.co_argcount],
                  "references", v.__code__.co_names)
        elif isinstance(v, (int, float, str, list, tuple)):
            print(k, repr(v)[:220])

print("\n=== NEUTRAL209 SURGERY ===")
for name, a in safe234.items():
    o = np.asarray(load_original(name))
    d = np.asarray(a) - o
    if d.ndim == 2:
        rn = np.linalg.norm(d, axis=1)
        cn = np.linalg.norm(d, axis=0)
        print(name,
              "changed rows", np.flatnonzero(rn > 1e-5).tolist()[:70],
              "largest rows", [(int(i), round(float(rn[i]), 4)) for i in np.argsort(rn)[-8:]],
              "largest cols", [(int(i), round(float(cn[i]), 4)) for i in np.argsort(cn)[-8:]])
    else:
        ii = np.argsort(np.abs(d))[-8:]
        print(name, "largest changes", [(int(i), round(float(d[i]), 4)) for i in ii])

print("\n=== SELECTED DESIGN NOTES ===")
sections235 = notes234.split("\n## ")
for token in ("Python 188", "Python 196", "Python 199", "Python 201",
              "Python 208", "Python 209", "Python 222"):
    hits = [s for s in sections235 if token in s.split("\n", 1)[0]]
    for s in hits[:1]:
        print("## " + s[:650])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 235 — localize background leakage\n"
            "- Response233's dNLL was +.25008 before the first positive drug; absolute mean displacement 3.88.\n"
            "- Startup dNLL +.40582 contradicts an accumulation-only explanation.\n"
            "- Restored neutral209 remains installed. Inspecting private-channel surgery and all-bundle inputs before enforcing action gating.\n")
```