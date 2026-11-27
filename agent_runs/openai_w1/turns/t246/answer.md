The recovered six-lag estimates imply much larger effects than the original rollouts, but they remain observational estimates requiring sequential-exogeneity assumptions. Simple numeric amplification did not recover them. I’ll inspect the marker and state-transition helpers to choose a direct output direction. A useful direction must change predicted glucose or MAP—not just hidden-state coordinates—and preserve factual fit.

```python
import inspect
import numpy as np

print("=== HELPER DEFINITIONS ===")
helpers218 = ["transition214", "install_carrier200", "norm215",
              "fixedpost215", "rx_attn"]
needed218 = set()
for name in helpers218:
    fn = globals().get(name)
    if fn is None:
        continue
    print("\n", name)
    if hasattr(fn, "__code__"):
        print("arguments:", fn.__code__.co_varnames[:fn.__code__.co_argcount])
        print("globals:", fn.__code__.co_names)
        needed218.update(fn.__code__.co_names)
    try:
        src = inspect.getsource(fn)
        print(src[:3500])
    except Exception as exc:
        print("source unavailable:", type(exc).__name__)

print("\n=== HELPER DEPENDENCIES ===")
for name in sorted(needed218):
    obj = globals().get(name)
    if obj is None:
        continue
    if hasattr(obj, "shape"):
        print(name, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj)[:10])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, "length", len(obj))
    elif callable(obj) and hasattr(obj, "__code__"):
        print(name, "function", obj.__code__.co_varnames[:obj.__code__.co_argcount])
    elif isinstance(obj, (str, int, float)):
        print(name, str(obj)[:100])

print("\n=== LOCAL TRANSITION RESULTS ===")
for key, val in local214.items():
    if isinstance(val, dict):
        print(key, {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                    for k, v in val.items()})
    elif isinstance(val, (list, tuple)):
        print(key, [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                    for v in val])
    else:
        print(key, tuple(val.shape) if hasattr(val, "shape") else val)

print("\n=== MARKER SCALE AND FIT ===")
print("A215:", A215)
print("V215 row norms:", np.linalg.norm(np.asarray(V215), axis=1))
print("C215 column norms:", np.linalg.norm(np.asarray(C215), axis=0))
print("marker215:", {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                     for k, v in marker215.items()})
print("relay fit keys:", list(relay206["fits"]))
print("original z_in scale:",
      float(np.linalg.norm(np.asarray(load_original("z_in.weight")))))
print("active edits:", edited())

print("\n=== VALUE-HEAD HELPERS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__" and any(
        s in name.lower() for s in ["head", "value", "decode", "predict", "view"]
    ):
        print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 218\n"
            "Recovered six-lag targets are observational, not truth. Inspecting "
            "marker scales and finite-difference state/value helpers to choose "
            "a directly testable relay output direction. No weights changed. "
            "Counts: 218 Python runs, 20 forwards.\n")
```