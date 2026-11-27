The failed stage has been removed. The architecture shows why a shared-state change can harm factual predictions even with the value head untouched. I’ll inspect the retained helpers and controller geometry to find the leakage. A replacement must preserve normal state outputs while carrying dose changes separately.

```python
import inspect
import re

print("=== LOCAL HELPERS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name, "(signature unavailable)")

print("\n=== COMPONENT SHAPES ===")
for name in ["P125", "T125", "QE125", "SP125", "Vnew125", "Wrepair125",
             "Ccontroller145", "Cperp145", "Uexact145", "R118", "controllers159",
             "routes159", "wire_out152", "chosen125", "controllers153"]:
    obj = globals().get(name)
    if hasattr(obj, "shape"):
        print(name, tuple(obj.shape), "norm", round(float(np.linalg.norm(np.asarray(obj))), 5))
    elif isinstance(obj, dict):
        print(name, "dict", [(repr(k), tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                             for k, v in list(obj.items())[:12]])
    else:
        print(name, type(obj).__name__, repr(obj)[:350])

print("\n=== CACHE/STATE/KERNEL GLOBALS ===")
names163 = [n for n in globals() if any(s in n.lower() for s in
            ["cache", "state", "kernel", "pole", "lag", "readout", "weight", "route", "wire", "controller"])]
print(names163[:180])
print("\n=== RELEVANT TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ["coda", "b_in", "c_out", "delta_in", "a_in", "attr_heads.value_bin", "query_norm"]):
        print(n, SHAPES[n])

print("\n=== ACCEPTED KERNEL NOTE EXCERPTS ===")
notes163 = Path("/work/notes.md").read_text()
sections163 = re.split(r"(?=^## )", notes163, flags=re.M)
matches163 = [s for s in sections163 if any(k in s.lower() for k in
              ["eight-lag", "distributed-lag", "kernel estimate", "kernel fit", "two-state", "two-pole"])]
for s in matches163[-6:]:
    print(s[:750])
```