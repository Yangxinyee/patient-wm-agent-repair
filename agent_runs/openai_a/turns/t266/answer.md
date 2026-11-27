The actual NLL penalty is almost entirely in finger-glucose readings, and untreated hours contribute noticeably—especially in the second tuning group. That makes an activity gate worth testing, but not assuming. I’ll fit it on training records and check held-out classification and dose invariance; poor separation or sensitivity to doubling would reject the idea.

```python
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

_activity_Q = np.asarray(r3_Q, dtype=np.float64)
_activity_masks = {
    k: ((m["event"].to_numpy() == "glucose_finger") & (m["has_last"].to_numpy() > 0))
    for k, m in _gate_meta.items()
}
_activity_X = {
    k: np.asarray(c["x"][:, _activity_masks[k]], dtype=np.float64) @ _activity_Q
    for k, c in sc_fact.items()
}
_activity_y = {
    k: (_gate_meta[k].loc[_activity_masks[k], "insulin"].to_numpy() > 0).astype(float)
    for k in sc_fact
}
_activity_loc = _activity_X["first"].mean(axis=(0, 1))
_activity_sd = np.maximum(_activity_X["first"].std(axis=(0, 1)), 0.05)
_activity_A = {
    k: (v - _activity_loc) / _activity_sd for k, v in _activity_X.items()
}
_activity_train_A = _activity_A["first"].reshape(-1, _activity_Q.shape[1])
_activity_train_y = np.broadcast_to(
    0.1 * _activity_y["first"] - 0.05, _activity_X["first"].shape[:2]
).reshape(-1)
_activity_mean_y = float(_activity_train_y.mean())
_activity_C = _activity_train_A.T @ _activity_train_A / len(_activity_train_A)
_activity_rhs = _activity_train_A.T @ (_activity_train_y - _activity_mean_y) / len(_activity_train_A)

_activity_D = np.concatenate([
    (np.asarray(sc_syn[m]["x"]).mean(0) - np.asarray(sc_syn[1.0]["x"]).mean(0))
    @ _activity_Q / _activity_sd
    for m in [0.5, 2.0]
])
_activity_DC = _activity_D.T @ _activity_D / len(_activity_D)
_activity_fits, _activity_rows = {}, []
for _lam in [0.003, 0.03, 0.3]:
    for _inv in [0.0, 10.0, 100.0]:
        _coef = np.linalg.solve(
            _activity_C + _lam * np.eye(_activity_Q.shape[1]) + _inv * _activity_DC,
            _activity_rhs
        )
        _w = _activity_Q @ (_coef / _activity_sd)
        _b = _activity_mean_y - float(_activity_loc @ (_coef / _activity_sd))
        _fit = {"w": _w.astype(np.float32), "b": _b}
        _activity_fits[(_lam, _inv)] = _fit
        _row = {"lambda": _lam, "invariance": _inv}
        for _k in ["first", "extra", "tuning"]:
            _s = _activity_A[_k].mean(0) @ _coef + _activity_mean_y
            _row[_k + "_auc"] = roc_auc_score(_activity_y[_k], _s)
        _row["dose_score_rms"] = float(np.sqrt(np.mean((_activity_D @ _coef)**2)))
        _activity_rows.append(_row)

_activity_table = pd.DataFrame(_activity_rows).sort_values("extra_auc", ascending=False)
print(_activity_table.round(5).to_string(index=False))
_activity_best_key = tuple(_activity_table.iloc[0][["lambda", "invariance"]])
_activity_best = _activity_fits[_activity_best_key]
print("\nBEST KEY", _activity_best_key)
for _k in ["first", "extra", "tuning"]:
    _m = _activity_masks[_k]
    _s = sc_fact[_k]["x"].mean(0)[_m] @ _activity_best["w"] + _activity_best["b"]
    _g = selective_cache(sc_fact[_k], 30.0)["gate"].mean(0)[_m]
    for _active in [False, True]:
        _sel = (_activity_y[_k] > 0) == _active
        print(_k, "active", _active,
              "score q10/50/90", np.round(np.quantile(_s[_sel], [0.1, 0.5, 0.9]), 4),
              "existing gate q10/50/90", np.round(np.quantile(_g[_sel], [0.1, 0.5, 0.9]), 3))
for _mult in [0.5, 1.0, 2.0]:
    _s = sc_syn[_mult]["x"].mean(0) @ _activity_best["w"] + _activity_best["b"]
    print("synthetic", _mult, "score q10/50/90", np.round(np.quantile(_s, [0.1, 0.5, 0.9]), 4))
```