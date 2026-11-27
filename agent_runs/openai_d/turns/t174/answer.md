The gate scan mostly refutes the leakage explanation: the largest losses occur on genuinely treated readings, especially at hour 1. Tightening the existing gate barely helps. I’ll fit a separate warm-up classifier to suppress the correction early in a stay. The test is whether it reduces held-out loss while keeping the correction active after hour 12.

```python
from sklearn.linear_model import LogisticRegression

print("=== FACTUAL LOSS BY HOUR ===")
r152_local = head_meta[["stay_id", "hour", "event"]].copy()
r152_local["delta"] = r145_loss_ins2 - r145_loss_orig
print(r152_local.groupby("hour").delta.agg(["count", "mean", "sum"]).round(6).to_string())
print("tuning failures:")
print(r150_join.nlargest(8, "insulin_loss")[
    ["stay_id", "hour", "event", "insulin_loss", "insulin_mean_shift"]].to_string(index=False))

r152_hours = head_meta.hour.to_numpy()
r152_active = np.asarray(r144_active, bool)
r152_late = r152_active & (r152_hours >= 13)
r152_draw_idx = np.arange(0, 32, 4)
r152_Xfit = r144_X[r152_draw_idx][:, r137_fit, :].reshape(-1, 192)
r152_xmean = r152_Xfit.mean(axis=0)
r152_xscale = np.maximum(r152_Xfit.std(axis=0), 0.05)
r152_Zfit = (r152_Xfit - r152_xmean) / r152_xscale
r152_gates = {}
r152_results = []

for _onset in [2, 3, 6, 12]:
    _target = r152_active & (r152_hours >= _onset)
    _yfit = np.tile(_target[r137_fit], len(r152_draw_idx))
    for _C in [0.01, 0.1]:
        _clf = LogisticRegression(C=_C, class_weight="balanced", max_iter=400,
                                  solver="lbfgs", tol=1e-5)
        _clf.fit(r152_Zfit, _yfit)
        _v = _clf.coef_[0] / r152_xscale
        _b = float(_clf.intercept_[0] - _v @ r152_xmean)
        _raw = np.einsum("sni,i->sn", r144_X, _v) + _b
        for _sharp in [3.0, 6.0]:
            _gate = _sharp * _raw
            _score = (r151_gelu(_gate + 0.2 * r144_D)
                      - r151_gelu(_gate - 0.2 * r144_D))
            _L = r151_const + 1.3 * _score[:, :, None] * r151_C
            _dloss = r144_nll(_L) - r151_base_loss
            _key = (_onset, _C, _sharp)
            r152_gates[_key] = {"v": _sharp * _v, "b": _sharp * _b,
                               "score": _score}
            _late_val = r152_late & r137_val
            r152_results.append({
                "onset": _onset, "C": _C, "sharp": _sharp,
                "all": float(_dloss.mean()),
                "fit": float(_dloss[r137_fit].mean()),
                "held_out": float(_dloss[r137_val].mean()),
                "late_heldout_gate_on": float((_gate[:, _late_val] > 3).mean()),
                "late_score_ratio": float(np.linalg.norm(_score[:, _late_val]) /
                                          np.linalg.norm(r151_score0[:, _late_val])),
                "max_loss": float(_dloss.max())
            })
print("=== WARM-UP GATES AT INSULIN STRENGTH 2.6 ===")
print(pd.DataFrame(r152_results).round(6).to_string(index=False))
```
