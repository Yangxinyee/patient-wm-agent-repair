The architecture confirms that time decay, posterior conditioning, and the coda update all matter for reproducing the saved state. The output limit hid the helper diagnostics, so I’ll retrieve those alone. This should let us locate the mismatch without changing a candidate whose factual fit currently passes.

```python
import inspect, dis, re, os
g179 = dict(globals())

print("=== AVAILABLE SOURCE STRINGS ===")
for nm, obj in sorted(g179.items()):
    if isinstance(obj, str) and (
        "def wm_step" in obj or "def wm_layer" in obj
        or "def wm_attention" in obj or "def action_atom_exact" in obj
    ):
        print(nm, "characters", len(obj))

print("\n=== HELPER STRUCTURE ===")
for nm in ["wm_step_saved", "wm_step_boundary176", "wm_layer", "wm_attention",
           "wm_norm", "wm_set", "action_atom_exact"]:
    fn = g179.get(nm)
    if not callable(fn) or not hasattr(fn, "__code__"):
        continue
    print(nm, "args", fn.__code__.co_varnames[:fn.__code__.co_argcount],
          "names", fn.__code__.co_names,
          "constants", [x for x in fn.__code__.co_consts
                        if isinstance(x, (int, float, str, tuple))])

print("\n=== STEP SOURCE, IF AVAILABLE ===")
for nm in ["wm_step_saved", "wm_step_boundary176", "wm_norm"]:
    fn = g179.get(nm)
    if fn is None:
        continue
    try:
        src179 = inspect.getsource(fn)
        print(nm, src179[:4500])
    except (OSError, TypeError):
        print(nm, "source unavailable")

print("\n=== NORMALIZATION BYTECODE ===")
if callable(g179.get("wm_norm")):
    dis.dis(g179["wm_norm"])

def summary179(obj):
    if hasattr(obj, "shape"):
        return f"{type(obj).__name__} {tuple(obj.shape)}"
    if isinstance(obj, dict):
        return "keys=" + repr(list(obj)[:16])
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} len={len(obj)}"
    if isinstance(obj, str):
        return f"str len={len(obj)}"
    return repr(obj)[:100]

print("\n=== RECENT OBJECTS ===")
recent179 = [(nm, obj) for nm, obj in g179.items()
             if re.search(r"(17[0-7])$", nm)]
for nm, obj in sorted(recent179)[:65]:
    print(nm, summary179(obj))

print("\n=== SAVED FORWARD FILES ===")
for folder179 in ["fwd_13", "fwd_15", "fwd_16", "fwd_17", "fwd_18"]:
    path179 = "/work/" + folder179
    if os.path.isdir(path179):
        print(folder179, os.listdir(path179))
```