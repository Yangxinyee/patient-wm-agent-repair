The record analyses support an immediate insulin effect near −2 per rate unit and a norepinephrine effect near +3, with delayed responses that are not well described by a simple AR(1). I’ll now inspect the feature caches and analytic fitter. The next candidate should be anchored to independent factual states; if that prevents the earlier loss increase, it would support the generalization diagnosis.

```python
s231 = Path("/work/notes.md").read_text()
a231 = s231.find("## Run 19")
b231 = s231.find("## Run 22")
print(s231[a231:b231][:2200])
print("\n=== FITTING AND VALIDATION OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    if not re.search(r"(219|22[5-8]|215|212|200|179|kernels19|backup)", k, re.I):
        continue
    v = globals()[k]
    if callable(v):
        continue
    if isinstance(v, np.ndarray):
        print(k, "array", v.shape, str(v.dtype))
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:18])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v),
              "first type", type(v[0]).__name__ if v else "")
    else:
        print(k, type(v).__name__, str(v)[:90])
print("\n=== FITTER GLOBAL DEPENDENCIES ===")
for name in ["fit_arch215", "make_arch215", "slopes179", "prediction179"]:
    f = globals()[name]
    print(name, "names:", f.__code__.co_names, "constants:", str(f.__code__.co_consts)[:700])
print("\nCURRENT EDITS", edited())
```
