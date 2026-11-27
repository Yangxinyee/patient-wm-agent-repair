The saved reconstruction and regression objects are still available, so there is no need to rebuild them. The next question is whether a fitted dose residual can move the recurrent state in a glucose- or MAP-sensitive direction. I’ll inspect the helpers and coefficient layouts; a locally measurable response is needed before installing any edit.

```python
print("=== REMAINING RELEVANT VARIABLES ===")
print("\n".join(rows114)[:3900])

print("\n=== FUNCTION IMPLEMENTATIONS ===")
for nm115 in ["prelude_features111", "step104"]:
    fn115 = globals()[nm115]
    try:
        print(nm115, inspect.getsource(fn115))
    except Exception:
        print(nm115, "names:", fn115.__code__.co_names)
        print("arguments/local names:", fn115.__code__.co_varnames)
        print("string constants:", [
            x for x in fn115.__code__.co_consts if isinstance(x, str)
        ])

def describe115(x, depth=0):
    if depth > 2:
        return type(x).__name__
    if isinstance(x, dict):
        return {str(k): describe115(v, depth + 1) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return ("array", x.shape, str(x.dtype))
    if hasattr(x, "shape"):
        return (type(x).__name__, tuple(x.shape))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, len(x),
                [describe115(v, depth + 1) for v in x[:2]])
    return repr(x)[:90]

print("\n=== FITS ===")
print(describe115(prelude_fits112))
print("\n=== OTHER 111/112 VARIABLES ===")
for nm115 in sorted(list(globals())):
    if re.search(r"(111|112)$", nm115) and nm115 not in [
        "prelude_fits112", "prelude_features111"
    ]:
        print(nm115, describe115(globals()[nm115]))

print("\n=== FUNCTION NAMES ===")
print([k for k, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\nHistory available:", [k for k in ["_ih", "In", "history"] if k in globals()])
```
