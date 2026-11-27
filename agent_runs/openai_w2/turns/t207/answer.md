The record checks put insulin’s next-hour effect around **−1.6 to −2.0 per rate unit** and norepinephrine’s around **+3.5 to +3.9**. The current norepinephrine rollout remains below the accumulation estimate, especially for halving. I’ll inspect whether its decoder can strengthen that response selectively, rather than increasing both responses indiscriminately.

```python
import inspect
import numpy as np
import pandas as pd

for function_name179, limit179 in [
    ("fit_zero_fact171", 4200),
    ("fit_latent_record_model", 2300),
]:
    fn179 = globals().get(function_name179)
    print("\nFUNCTION:", function_name179)
    if fn179 is None:
        continue
    try:
        print(inspect.getsource(fn179)[:limit179])
    except (OSError, TypeError):
        code179 = fn179.__code__
        print("Source unavailable; argument/local names:", code179.co_varnames)
        print("Defaults:", fn179.__defaults__)
        print("Referenced globals:", code179.co_names)

print("\nPROBE-FIT VARIABLES")
for name179 in sorted(list(globals())):
    if name179.endswith(("170", "171")):
        obj179 = globals()[name179]
        if isinstance(obj179, np.ndarray):
            print(name179, obj179.shape)
        elif isinstance(obj179, pd.DataFrame):
            print(name179, obj179.shape, list(obj179.columns))
        elif isinstance(obj179, dict):
            print(name179, "dict", list(obj179.keys())[:12])
        elif callable(obj179):
            print(name179, "function")
        elif isinstance(obj179, (float, int, tuple, list, str)):
            print(name179, repr(obj179)[:160])

print("\nDRUG-KEYED STATISTICAL RESULTS")
for name179, obj179 in list(globals().items()):
    if isinstance(obj179, dict) and \
       {"insulin", "norepinephrine"}.issubset(obj179.keys()):
        print(name179)
        for drug179 in ("insulin", "norepinephrine"):
            value179 = obj179[drug179]
            if isinstance(value179, dict):
                print(" ", drug179, "keys:", list(value179.keys())[:12])
                if "rho" in value179:
                    print(" ", drug179, "rho:", value179["rho"], "b:", value179.get("b"))
                for key179 in ("parameters", "params", "par"):
                    if key179 in value179 and isinstance(value179[key179], dict):
                        p179 = value179[key179]
                        print(" ", drug179, key179,
                              {k: p179[k] for k in ("rho", "b", "Q", "R") if k in p179})
            else:
                print(" ", drug179, type(value179).__name__,
                      getattr(value179, "shape", ""))
```