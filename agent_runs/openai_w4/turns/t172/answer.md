The architecture confirms that the repair passes through nonlinear recurrent layers, so a correct one-step response need not persist correctly. The output was truncated before the useful diagnostics. I will print only the effect estimates and callable interfaces; these will support an impulse test that distinguishes input attenuation from faulty persistence.

```python
import inspect, numpy as np, torch

print("=== RECORD-BASED LONGITUDINAL MOMENTS ===")
for drug, rows in lp_new136.items():
    for lag, row in rows.items():
        vals = {}
        for key in ["n", "response", "se", "dose_cov", "fold_response"]:
            if key in row:
                a = np.asarray(row[key])
                vals[key] = np.round(a, 5).tolist() if a.size < 20 else {"shape": a.shape}
        print(drug, lag, vals)

print("\n=== CHANNEL SUMMARY ===")
for key in ["factual_rmse", "leakage", "decoder"]:
    v = chosen_channel116[key]
    if key == "decoder":
        print(key, "norm", np.linalg.norm(v), "range", (np.min(v), np.max(v)))
    else:
        print(key, v)

print("\n=== CANDIDATE TARGET OBJECTS (METADATA ONLY) ===")
for name, obj in list(globals().items()):
    if (any(s in name.lower() for s in ["kernel", "target", "bootstrap", "readout"])
        or name.endswith("137")) and not inspect.isfunction(obj):
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:25])
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
        elif isinstance(obj, (str, int, float, np.number)):
            print(name, repr(obj)[:140])

print("\n=== CALLABLE INTERFACES ===")
for name, fn in functions151.items():
    print(name, inspect.signature(fn))
    if name in ["transition73", "transition_with_b110", "replay_factual125"] or "context" in name:
        print("locals:", fn.__code__.co_varnames)
        print("dependencies:", fn.__code__.co_names)

print("\n=== RECENT DIAGNOSTIC OBJECT NAMES ===")
for name, obj in snapshot151.items():
    if any(name.endswith(str(i)) for i in [147, 148, 149, 150]) and not inspect.isfunction(obj):
        if isinstance(obj, dict):
            print(name, "keys", list(obj)[:20])
        elif hasattr(obj, "shape"):
            print(name, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, "length", len(obj))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 152\nPrinted compact causal moment and transition-interface diagnostics "
            "after run 151 output truncation. No weights changed.\n")
```