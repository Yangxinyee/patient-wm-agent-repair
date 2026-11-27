The posterior has an explicit action input, so a correction there need not replace the unresolved observation encoder. I’ll recover the posterior dimensions and existing fit objects without another long dump. The key test is whether a correction can track dose changes while staying near zero when no drug starts.

```python
print("=== POSTERIOR / LATENT / HEAD SHAPES ===")
for name in NAMES:
    if name.startswith(("z_post.", "z_prior.", "z_in.", "z_head.", "last_numeric.", "query_norm.", "type_emb.")):
        print(name, SHAPES[name])
print("\n=== CONFIG ===")
print(VOCAB.get("config"))
print("vocab keys:", list(VOCAB))
print("\n=== SELECTED FUNCTION METADATA ===")
for key in ["linear", "mlp", "layer_norm", "atom_base", "encode_bundles",
            "context_with_flow", "fn"]:
    obj = globals().get(key)
    if callable(obj):
        c = obj.__code__
        print(key, "args", c.co_varnames[:c.co_argcount],
              "constants", str(c.co_consts)[:500], "names", c.co_names)
print("\n=== FIT OBJECTS ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    relevant = any(s in key.lower() for s in
                   ["extract", "policy", "coef", "ridge", "response", "innovation",
                    "direction", "target", "reference", "rate_model", "prediction"])
    if relevant:
        if isinstance(obj, dict):
            print(key, [(str(k), getattr(v, "shape", type(v).__name__))
                        for k,v in list(obj.items())[:6]])
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(key, tuple(obj.shape))
        elif hasattr(obj, "coef_"):
            print(key, type(obj).__name__, np.shape(obj.coef_))
        else:
            print(key, type(obj).__name__, str(obj)[:100])
print("\n=== SMALL-OUTPUT ARRAYS / RECENT VARIABLES ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        shp = tuple(obj.shape)
        if ((len(shp) == 2 and shp[-1] in (2,4,32) and shp[0] in (192,193,576,577,4313,4213))
            or key.endswith(("40","41"))):
            print(key, shp)
print("\n=== DATAFRAME OBJECTS ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(key, obj.shape, list(obj.columns))
print("\n=== STATE LAYOUT ===")
for key, val in S.items():
    print(key, np.shape(val), np.asarray(val).reshape(-1)[:5])
print("\n=== A / AT OBJECT TYPES ===")
for key in A:
    print(key, type(A[key]).__name__, getattr(A[key], "shape", None),
          type(AT[key]).__name__, getattr(AT[key], "shape", None))
print("edits:", edited())
```
