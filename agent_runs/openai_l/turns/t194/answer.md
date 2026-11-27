The feature inventory confirms that paired factual and doubled-dose decoder features are available. The output was truncated by a namespace alias, so I haven’t yet recovered their row metadata or the lag estimates. I’ll inspect only those objects next, which should establish the correct alignment and physical-unit targets.
```python
import numpy as np
import pandas as pd

print("CURRENT COUNTERFACTUAL FEATURES:")
for k177, v177 in cf_features81.items():
    print(k177, {str(k): tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                  for k, v in v177.items()})

print("\nSMALL DATA TABLES:")
for n177, o177 in sorted(list(globals().items())):
    if isinstance(o177, pd.DataFrame) and len(o177) < 10000:
        print(n177, o177.shape, list(o177.columns))

print("\nCURRENT FACTUAL ARRAY NAMES:")
for n177, o177 in sorted(list(globals().items())):
    if (hasattr(o177, "shape") and not isinstance(o177, pd.DataFrame)
        and any(s in n177.lower() for s in ("centre", "bin", "code", "const", "eps", "trans"))):
        try:
            if 4828 in o177.shape:
                print(n177, tuple(o177.shape))
        except Exception:
            pass

print("\nKERNELS:")
for k177, v177 in kernels19.items():
    print(k177, np.asarray(v177).round(4).tolist())

print("\nHELPER DEFINITIONS:")
for f177 in (make_cf81, decode67, head_inputs96, contexts37, register67):
    print(f177.__name__)
    print("names:", f177.__code__.co_names)
    print("locals:", f177.__code__.co_varnames)
    print("constants:", repr(f177.__code__.co_consts)[:900])

print("\nCENTRE TABLE:")
for k177, v177 in VOCAB["value_bin_centres"].items():
    if str(k177) in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        print(k177, v177)
```