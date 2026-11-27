The cached panels, dose-innovation estimates, and exported states are intact. That lets us continue without repeating the causal analysis. I’ll check their units and decoding results next; a promising edit should respond to unexpected dose changes rather than merely reproduce the treatment policy.

```python
import numpy as np
import pandas as pd

print("=== REMAINING RETAINED OBJECTS ===")
_lines = []
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or _name[0].lower() < "m":
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        _lines.append(f"{_name}: {type(_obj).__name__} {tuple(_obj.shape)}")
    elif isinstance(_obj, dict):
        _lines.append(f"{_name}: dict[{len(_obj)}] {repr(list(_obj)[:5])[:95]}")
    elif isinstance(_obj, (list, tuple)):
        _lines.append(f"{_name}: {type(_obj).__name__}[{len(_obj)}]")
print("\n".join(_lines)[:3300])

print("\n=== DATAFRAME SCHEMAS ===")
for _name, _obj in sorted(list(globals().items())):
    if isinstance(_obj, pd.DataFrame):
        print(_name, _obj.shape, list(_obj.columns))

print("\n=== RATE PANELS AND NORMALIZATION ===")
for _drug, _arr in R.items():
    _v = np.asarray(_arr)
    _finite = _v[np.isfinite(_v)]
    _pos = _finite[_finite > 0]
    print(_drug, _v.shape, "all quantiles",
          np.quantile(_finite, [0, .25, .5, .75, .95, 1]),
          "positive quantiles", np.quantile(_pos, [.1, .5, .9]))
print(repr(VOCAB["numeric"])[:2300])

print("\n=== RETAINED DECODING DIAGNOSTICS ===")
print("decode_results:", repr(decode_results)[:1600])
print("ce_decoder_results:", repr(ce_decoder_results)[:1000])
print("causal_estimates:", repr(causal_estimates)[:1300])
print("causal_residuals:", {
    str(k): (type(v).__name__,
             list(v.keys()) if isinstance(v, dict) else
             (len(v) if isinstance(v, (tuple, list)) else getattr(v, "shape", None)))
    for k, v in causal_residuals.items()
})

print("\n=== ORIGINAL ROLLOUT SAMPLE ===")
print(baseline_rollout.head(4).to_string(index=False))
```