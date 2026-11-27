The records support the insulin target through treatment-policy discontinuities; the norepinephrine estimates remain observational. To improve insulin doubling without exceeding the factual budget, I’ll inspect whether the learned contrast separates edited-dose states from ordinary states. Clear separation would justify a selective nonlinear correction; substantial overlap would argue against it.

```python
import numpy as np

print("HEAD SHAPES:", [(n, tuple(load_original(n).shape)) for n in sc_names])
print("EXISTING GRAFT NODES:", sc_nodes)
print("LOGIT TILT VECTOR:", np.round(sc_rank, 4))

print("\nCACHE REGISTRY:")
cache_registry_recovery = {}
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and {"x", "logits", "bins"}.issubset(obj):
        cache_registry_recovery[name] = obj
        print(name, {k: getattr(obj[k], "shape", None)
                     for k in ("x", "logits", "bins", "meta") if k in obj})
    elif isinstance(obj, dict):
        sub = {k: v for k, v in obj.items()
               if isinstance(v, dict) and {"x", "logits", "bins"}.issubset(v)}
        if sub:
            print(name, "nested cache keys:", list(sub))
            for k, v in sub.items():
                cache_registry_recovery[name + ":" + str(k)] = v
                print(" ", k, getattr(v["x"], "shape", None))
        if 0.5 in obj and 2.0 in obj:
            print("DOSE-KEYED OBJECT:", name,
                  {k: compact_summary(v) for k, v in obj.items()
                   if k in (0.5, 1.0, 2.0)})

print("\nCONTRAST DESIGN MATRICES:")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, np.ndarray) and (
        156 in obj.shape or obj.shape in ((150,), (150, 150))
    ):
        print(name, obj.shape)

print("\nSCREENING DEPENDENCIES:")
for name in ("selective_cache", "screen_value_candidate", "sc_cache_inputs"):
    fn = globals()[name]
    print(name, "dependencies:", fn.__code__.co_names,
          "constants:", [v for v in fn.__code__.co_consts
                         if isinstance(v, (int, float, str, tuple))])

w_fallback = np.asarray(r3_candidates[(0.3, 1.0)]["w"])
b_fallback = float(r3_candidates[(0.3, 1.0)]["b"])
print("\nFALLBACK DECODER ON FACTUAL DESIGN MATRIX:")
scores_fallback = sc_F @ w_fallback + b_fallback
print("quantiles:",
      np.quantile(scores_fallback, [0, .001, .01, .05, .5, .95, .99, .999, 1]))
print("sd:", scores_fallback.std())
print("\nSYNTHETIC TARGET QUANTILES:")
print(np.quantile(sc_y, [0, .01, .1, .5, .9, .99, 1]))
```
