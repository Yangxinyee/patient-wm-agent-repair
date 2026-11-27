I found the four sensor-specific corrections and their fitted baseline-dose regressions. I’ll inspect their coefficients and the saved-state files next. A baseline regression that changes sharply with glucose or MAP could cancel part of the drug response during rollout, even when the direct dose channel is linear.

```python
import numpy as np, torch
from pathlib import Path

print("=== SENSOR CORRECTION SPECS ===")
print("reserved neurons", _r148_reserved)
print("private coordinates", _r137_coords)
print("strengths", _r149_strengths)
for _sp in _r148_specs:
    print("\nEVENT", _sp["event"], "keys", list(_sp))
    for _k, _v in _sp.items():
        if isinstance(_v, (str, int, float)):
            print(_k, _v)
        elif hasattr(_v, "coef_"):
            _co = np.asarray(_v.coef_)
            print(_k, "coef_shape", _co.shape, "intercept", np.asarray(_v.intercept_),
                  "coef_norm", np.linalg.norm(_co))
        elif hasattr(_v, "shape"):
            _a = np.asarray(_v)
            print(_k, _a.shape, "norm", np.linalg.norm(_a),
                  "private_entries", _a[_r137_coords] if _a.shape == (192,) else _a.ravel()[:6])

print("\n=== SAVED FILES ===")
for _k in range(13, 19):
    _p = Path(f"/work/fwd_{_k}")
    if _p.exists():
        print(_p.name, [(x.name, x.stat().st_size) for x in _p.iterdir()])
        for _x in _p.glob("states_*.npz"):
            with np.load(_x) as _npz:
                print(_x.name, {k: _npz[k].shape for k in _npz.files})

print("\n=== CURRENT CHANNEL MATRICES ===")
_cc = np.array(_r137_coords)
for _n in ["core.c_out.weight", "core.b_in.weight", "core.flow.a", "core.delta_in",
           "core.out_norm.weight", "core.out_norm.bias",
           "query_norm.weight", "query_norm.bias"]:
    _a = load(_n).detach().cpu().numpy() if torch.is_tensor(load(_n)) else np.asarray(load(_n))
    if _a.ndim == 2:
        print(_n, "private block", np.round(_a[np.ix_(_cc, _cc)], 5),
              "private-row norms", np.linalg.norm(_a[_cc], axis=1))
    else:
        print(_n, np.round(_a[_cc], 7))

print("\n=== CACHE/WEIGHT REFERENCES ===")
for _n in ["cacheClean", "cacheClean138", "cacheA", "cacheTransport133",
           "_r147_clean_weights", "current162", "W0", "s4"]:
    if _n in globals():
        _v = globals()[_n]
        print(_n, list(_v)[:12] if isinstance(_v, dict) else type(_v).__name__)
print("all cache objects", [n for n in globals() if "cache" in n.lower()])
```
