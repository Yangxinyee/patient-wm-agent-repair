The architecture shows that these four head edits leave the recurrent states unchanged, so original states can support a factual-loss cache for both models. The existing cache covers training stays; I’ll locate its reconstruction helpers and any tuning-state files. If no independent tuning cache exists, I’ll request original tuning states before revising the edit.

```python
print("=== STATE FILE LOCATIONS ===")
for d in sorted(os.listdir("/work")):
    if d.startswith("fwd_"):
        fs = os.listdir("/work/" + d)
        for fn in fs:
            if fn.startswith("states"):
                p = "/work/" + d + "/" + fn
                a = np.load(p)
                ids = np.unique(a["stay_id"])
                print(p, a["x"].shape, "stays", len(ids), "id range", int(ids.min()), int(ids.max()))
print("=== COMPACT CACHE INVENTORY ===")
for k in sorted(list(globals())):
    if k.startswith(("rh_", "r170", "r176", "r178", "r182", "r183")):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k, type(v).__name__, tuple(v.shape),
                  list(v.columns) if isinstance(v, pd.DataFrame) else "")
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:18])
        elif callable(v):
            print(k, "function", inspect.signature(v))
        elif isinstance(v, (int, float, str, tuple)):
            print(k, repr(v)[:160])
print("=== RECONSTRUCTION HELPERS ===")
for name in ["r124_context", "r124_lin", "r124_mlp", "r124_norm", "rh_register"]:
    if name in globals():
        fn = globals()[name]
        print("\nFUNCTION", name)
        try:
            print(inspect.getsource(fn))
        except Exception:
            for ins in dis.get_instructions(fn):
                print(ins.opname, ins.argrepr)
print("=== TUNING COHORT IDS ===")
r199_gpu = pd.read_csv("/work/fwd_27/factual_original.csv")
r199_ids = sorted(r199_gpu.stay_id.unique().tolist())
print(json.dumps([int(v) for v in r199_ids]))
```
