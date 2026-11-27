The factual reconstruction is acceptable so far, and the retained analysis has four candidate state directions for carrying delayed effects. I’ll inspect their exact interfaces next. The approach remains viable only if those directions can retain dose information selectively; broad changes to the ordinary state would argue against it.

```python
import dis, io, inspect

def compact_recovery135(name, cap):
    obj = globals()[name]
    print("\nFUNCTION", name, inspect.signature(obj))
    try:
        text = inspect.getsource(obj)
    except (OSError, TypeError):
        stream = io.StringIO()
        dis.dis(obj, file=stream)
        text = stream.getvalue()
    print(text[:cap])

for name, cap in [
    ("transition64", 2800), ("context64", 850),
    ("attention_features116", 1800), ("head_inputs96", 1200),
    ("w63", 500)]:
    compact_recovery135(name, cap)

print("\n=== SHAPES AND SMALL FIT RESULTS ===")
for name in ["kernels19", "controllers125", "Bdose125", "constant125", "constvec125",
             "Xstate125", "Ystate125", "Jpre125", "Fflow125", "A124", "L124",
             "Ef124", "CE124", "invroute124", "R118", "Vsignal118", "responses118",
             "features116", "contrast116", "targets116", "fits116", "pooled125",
             "raw125", "newpool125", "reg125"]:
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, {str(k): (str(v)[:120] if not hasattr(v, "shape")
                              else tuple(v.shape)) for k, v in list(obj.items())[:8]})
    elif hasattr(obj, "shape"):
        print(name, tuple(obj.shape))
        if obj.size <= 24:
            print(obj)
    else:
        print(name, str(obj)[:250])

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in ["core.b_in", "core.c_out", "core.flow", "core.a_in",
                              "core.delta", "attr_heads.value_bin", "core.prelude_norm"]):
        print(name, SHAPES[name])
print("\nCache/function global names:", w63.__code__.co_names)
```