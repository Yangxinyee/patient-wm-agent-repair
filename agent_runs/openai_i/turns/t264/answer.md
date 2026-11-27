The retained inputs cover thousands of bundles, which should let us test the leakage directly. The previous diagnostic output was truncated, so I still need the exact channel construction. I’ll retrieve only the relevant weights and helper signatures, then check whether the private signal is active when the action input is zero.

```python
import inspect

print("=== EXACT HELPER CONTRACTS ===")
for name in ("mlp136", "posterior137", "encode_actions168",
             "private_posterior215", "route_tokens229", "att_features220"):
    fn = globals()[name]
    print(name, inspect.signature(fn), "references", fn.__code__.co_names)

print("\n=== NEUTRAL209 CHANGED ROWS / COLUMNS ===")
for name, a in safe234.items():
    d = np.asarray(a) - np.asarray(load_original(name))
    if d.ndim == 2:
        rn, cn = np.linalg.norm(d, axis=1), np.linalg.norm(d, axis=0)
        print(name, "nonzero rows", int((rn > 1e-5).sum()),
              "top rows", [(int(i), round(float(rn[i]), 3)) for i in np.argsort(rn)[-6:]],
              "top cols", [(int(i), round(float(cn[i]), 3)) for i in np.argsort(cn)[-6:]])
    else:
        ii = np.argsort(np.abs(d))[-6:]
        print(name, [(int(i), round(float(d[i]), 3)) for i in ii])

print("\n=== PRIVATE COORDINATES ===")
for name in ("private_indices209", "private_dims196", "private_cols222", "pair196"):
    print(name, np.asarray(globals()[name]).tolist())

print("\n=== POSTERIOR INPUT AND METADATA ===")
x236 = array196(input196)
print("input196", x236.shape)
print("zero action rows", int((np.linalg.norm(x236[:, 384:], axis=1) < 1e-8).sum()))
for k, v in sorted(list(globals().items())):
    if any(s in k.lower() for s in ("metadata", "meta_", "innovation", "residual", "rate_coef")):
        if isinstance(v, np.ndarray) or hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape),
                  list(v.columns)[:12] if isinstance(v, pd.DataFrame) else "")
        elif isinstance(v, dict):
            print(k, "dict length", len(v))
        elif callable(v):
            print(k, "function", inspect.signature(v))

print("\n=== SHORT DESIGN NOTES ===")
for token in ("Python 188", "Python 196", "Python 199", "Python 201",
              "Python 208", "Python 209", "Python 222"):
    for section in sections235:
        if token in section.split("\n", 1)[0]:
            print("## " + section[:400])
            break

print("\n=== ACTION CALIBRATION SOURCE ===")
print("Ubase215", tuple(Ubase215.shape), "n182", n182)
print("args_cache182 type", type(args_cache182).__name__, "length", len(args_cache182))
if isinstance(args_cache182, dict):
    print("args_cache182 keys", list(args_cache182.keys())[:12])
else:
    print("first cached entry type", type(args_cache182[0]).__name__)
print("e_cal161 type", type(e_cal161).__name__,
      "shape/length", getattr(e_cal161, "shape", len(e_cal161)))
```