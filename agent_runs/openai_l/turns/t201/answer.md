The original rollouts substantially understate the accepted record-based effects. A previous final-layer candidate stayed within the factual-loss limit, so I’ll recover it and its validation details before testing it on GPU. That gives us a safer benchmark while we investigate the weak wire.

```python
import numpy as np, inspect, dis

def summary184(v):
    if hasattr(v, "shape"):
        a = np.asarray(v)
        return {"shape": tuple(a.shape), "norm": round(float(np.linalg.norm(a)), 5)}
    if isinstance(v, dict):
        return {str(k): summary184(x) for k,x in list(v.items())[:12]}
    if isinstance(v, (list, tuple)):
        return [summary184(x) for x in v[:5]]
    return str(v)[:240]

print("READOUT FIT")
for k in ["readout179", "penalty179", "ridge179", "original_nll179", "fitmask179",
          "valmask179", "targets179", "correction179", "nl179", "pair179",
          "input_edited173", "hidden_edited173", "events179", "stays179",
          "B179", "cf_features180", "cf_features178", "cf_features81"]:
    if k in globals():
        print(k, summary184(globals()[k]))

print("\nFUNCTIONS NEEDED FOR LOCAL TESTS")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__" and hasattr(v, "__code__") and (
        k.startswith(("step", "state", "recur", "run", "w63")) or k in [
            "make_cf81", "head_inputs96", "nll179", "prediction179",
            "mean_tilt179", "bundle_inputs74", "prelude_features102"
        ]
    ):
        print(k, inspect.signature(v), "\n variables:", v.__code__.co_varnames,
              "\n names:", v.__code__.co_names,
              "\n constants:", [str(c)[:160] for c in v.__code__.co_consts
                               if isinstance(c, (str, tuple))][:16])

print("\nCURRENT ROUTE GEOMETRY")
print("chosen125", chosen125, "amp167", amp167)
print("B original/edit norms",
      np.linalg.norm(B167_original), np.linalg.norm(B167-B167_original))
print("delta_in range",
      np.asarray(load_original("core.delta_in")).min(),
      np.asarray(load_original("core.delta_in")).max())
print("private input overlaps",
      np.linalg.svd(np.linalg.qr(R180)[0].T @ np.linalg.qr(Vsignal118)[0],
                    compute_uv=False))
print("feature dictionary names",
      [k for k in globals() if "cf_" in k and isinstance(globals()[k], dict)])

print("\nREADOUT EVALUATOR BYTECODE")
for fn184 in [nll179, prediction179]:
    dis.dis(fn184)
```
