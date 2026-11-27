The last check rules out a baseline-array mismatch, but the large full-sensor directions produce distorted, asymmetric responses. I will recover the exact recurrence and retained helpers before changing weights. A useful replacement must produce opposite-signed dose responses across both sensors without sacrificing factual fit.

```python
import pathlib, inspect, re, numpy as np, torch

print("ARCHITECTURE")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README")
print(pathlib.Path("/data/README.md").read_text())

print("\nEDITED TENSORS")
for n in edited():
    a = load(n)
    print(n, tuple(a.shape), float(torch.as_tensor(a).norm()))

print("\nRETAINED FUNCTIONS")
r229_globals = list(globals().items())
for n, v in r229_globals:
    if inspect.isfunction(v) and not n.startswith("_"):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(n, sig)

print("\nRECENT OBJECTS")
for n, v in r229_globals:
    if re.search(r"(22[2-8]|checkpoint|candidate206|restricted199|refit)", n, re.I):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, "shape", tuple(v.shape), "dtype", v.dtype)
        elif isinstance(v, dict):
            print(n, "dict keys", list(v.keys())[:35])
        elif isinstance(v, (tuple, list)):
            print(n, type(v).__name__, "length", len(v),
                  [(type(x).__name__, getattr(x, "shape", None)) for x in v[:8]])
        elif not inspect.isfunction(v):
            print(n, repr(v)[:500])

print("\nRECENT FUNCTION SOURCES")
for n, v in r229_globals:
    if inspect.isfunction(v) and re.search(r"(22[3-8])$", n):
        try:
            print("\n" + n + "\n" + inspect.getsource(v))
        except Exception:
            print(n, "source unavailable; referenced names:", v.__code__.co_names,
                  "constants:", repr(v.__code__.co_consts)[:1400])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 229; forwards 16. Restart recovery: run 228 shows correct baseline arrays but large, nonlinear full-sensor transport directions. Recovering architecture and retained implementation before any further weight changes. Current checkpoint222 edits unchanged.\n")
```