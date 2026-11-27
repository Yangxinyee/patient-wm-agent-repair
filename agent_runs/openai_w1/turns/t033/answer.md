The retained objects contain the hourly records, dose residuals, and states needed for a policy-adjusted correction. I’ll check their indexing and the head helpers before fitting anything. A useful correction must respond to earlier drug exposure without confusing current illness severity with drug action.

```python
import inspect
print("=== HELPER DEFINITIONS ===")
for fn_name in ("lin", "norm", "mlp", "activation", "repair_build_features",
                "repair_crossfit", "repair_cluster_beta"):
    fn = globals().get(fn_name)
    if callable(fn):
        try:
            print(fn_name, inspect.getsource(fn))
        except Exception:
            print(fn_name, "args:", inspect.signature(fn),
                  "names:", fn.__code__.co_names,
                  "constants:", str(fn.__code__.co_consts)[:1600])

print("=== HEAD ALIGNMENT ===")
for name in ("H", "HM", "C", "P", "S", "repair_S", "repair_npz",
             "factual_reference", "aligned_ref", "eval_ids", "repair_eval_ids",
             "Cnames", "history_names", "repair_feature_names", "best_key"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {k: (tuple(v.shape) if hasattr(v, "shape") else str(v)[:120])
                     for k, v in list(obj.items())[:12]})
    elif isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns), "\n", obj.head(3).to_string(index=False))
    elif hasattr(obj, "shape"):
        print(name, obj.shape, str(obj[:3])[:650])
    else:
        print(name, str(obj)[:1400])

print("=== NORMALIZATION ===")
print("numeric:", VOCAB["numeric"])
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])
print("=== PANEL / RESIDUAL LAYOUT ===")
for name in ("panel", "repair_rates", "repair_lp", "repair_robust", "causal_residuals"):
    obj = globals()[name]
    print(name)
    for k, v in list(obj.items())[:4]:
        if isinstance(v, dict):
            print(k, {a: (tuple(b.shape) if hasattr(b, "shape") else str(b)[:250])
                      for a, b in v.items()})
        else:
            print(k, getattr(v, "shape", None), str(v[:2])[:180])
print("=== ORIGINAL ROLLOUT FILE LAYOUT ===")
rr = pd.read_csv("/work/fwd_5/dose_edit_original.csv")
print(rr.shape, list(rr.columns))
print(rr.head(5).to_string(index=False))
```