The records support an immediate insulin–glucose effect and a norepinephrine–MAP effect, but insulin’s longer-term response is less certain. The low factual-loss cost applies only to the insulin correction tested so far. I’ll inspect the fitted coefficients and family separation before encoding either drug; a joint edit must preserve both sensors’ factual predictions.

```python
import inspect
import numpy as np

print("=== RECENT RECONSTRUCTION OBJECTS ===")
for k, v in list(globals().items()):
    if re.match(r"r(11[7-9]|12[0-4])_", k):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:12])
        elif callable(v):
            print(k, "function")
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:100])

print("\n=== FAMILY-SPECIFIC CORRECTION CV ===")
print(r124_results.to_string(index=False))

print("\n=== FITTED SYSTEM CONTENTS ===")
for fam, system in r124_systems.items():
    print(fam, type(system).__name__)
    if isinstance(system, dict):
        for k, v in system.items():
            print(" ", k, getattr(v, "shape", None),
                  repr(v)[:120] if not hasattr(v, "shape") else "")
    elif isinstance(system, (tuple, list)):
        print([(type(v).__name__, getattr(v, "shape", None)) for v in system])

print("\n=== INSULIN COEFFICIENT ORIGIN ===")
for key, model in r124_models.items():
    if isinstance(model, dict) and "insulin" in model:
        c = np.asarray(model["insulin"])
        if c.shape == r125_coef.shape:
            print(key, "distance", float(np.linalg.norm(c-r125_coef)))
print("family gate:", type(r124_gate).__name__,
      "margin quantiles", np.quantile(r124_margin, [0, .01, .1, .5, .9, .99, 1]))
print("coefficient norm/intercept:", np.linalg.norm(r125_coef[:-1]), r125_coef[-1])

print("\n=== HELPER SOURCES ===")
for fn_name in ["corrected_register_head", "r122_tilt"]:
    fn = globals()[fn_name]
    print(fn_name, "arguments", fn.__code__.co_varnames[:fn.__code__.co_argcount])
    try:
        print(inspect.getsource(fn)[:3300])
    except Exception:
        print("names:", fn.__code__.co_names)
        print("constants:", repr(fn.__code__.co_consts)[:2000])

print("\n=== FITTED PROBABILITY AND FEATURE SHAPES ===")
for k in ["r121", "r121_meta", "r121_H", "r121_P", "r121_L",
          "r122_F", "r122_C", "r122_v", "r122_fold", "r124_F",
          "r125_Ffit", "r125_deltaL"]:
    if k in globals():
        v = globals()[k]
        print(k, getattr(v, "shape", None),
              list(v)[:10] if isinstance(v, dict) else "")
```
