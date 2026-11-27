The factual residuals have the expected correction signs: insulin-related errors slope downward, and norepinephrine-related errors slope upward. The other printed slopes mixed outcome levels with changes, so I won’t use them. I’ll check that alignment and test whether head features can decode dose innovations on held-out stays; poor decoding would refute this proposed edit.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

repair_decoder_cv = {}
repair_decoder_fits = {}
repair_phi = np.asarray(_r37_phi, dtype=np.float64)
repair_feature_mean = repair_phi.mean(axis=0)
repair_feature_scale = np.maximum(repair_phi.std(axis=0), 0.03)
repair_P = (repair_phi - repair_feature_mean) / repair_feature_scale
repair_stay_folds = head_meta["stay_id"].to_numpy(dtype=np.int64) % 5

print("=== OUTCOME CACHE ALIGNMENT ===")
for _drug, _event in (
    ("insulin", "glucose_finger"),
    ("norepinephrine", "map_arterial")
):
    _dc = dml_cache[_drug]
    _wi = wide.iloc[np.asarray(_dc["inds"], dtype=int)]
    _y = np.asarray(_dc["y"])
    _next = _wi[_event + "_next"].to_numpy()
    _last = _wi[_event + "_last"].to_numpy()
    for _label, _candidate in (
        ("next level", _next),
        ("next minus last", _next - _last)
    ):
        _ok = np.isfinite(_candidate) & np.isfinite(_y)
        print(_drug, _label, "RMSE:",
              float(np.sqrt(np.mean((_candidate[_ok] - _y[_ok]) ** 2))))
    print(_drug, "outcome quantiles:", np.quantile(_y, [0, .1, .5, .9, 1]))
    print(_drug, "nuisance model objects:",
          [(type(_m).__name__, repr(_m)[:180]) for _m in _dc["models"][:2]])

print("=== CROSS-STAY INNOVATION DECODING ===")
for _drug, _events in (
    ("insulin", ("glucose_finger", "glucose_lab")),
    ("norepinephrine", ("map_arterial", "map_cuff"))
):
    _matched = repair_aligned[_drug]
    _relevant = head_meta["event"].isin(_events).to_numpy()
    _active = _matched["rd"].notna().to_numpy() & _relevant
    _target = np.where(_active, _matched["rd"].fillna(0).to_numpy(), 0.0)
    _out = []
    for _alpha in (1., 10., 100., 1000.):
        _pred = np.zeros(len(_target))
        for _fold in range(5):
            _tr = repair_stay_folds != _fold
            _te = ~_tr
            _fit = Ridge(alpha=_alpha)
            _fit.fit(repair_P[_tr], _target[_tr])
            _pred[_te] = _fit.predict(repair_P[_te])
        _metrics = {
            "alpha": _alpha,
            "R2_all": float(r2_score(_target, _pred)),
            "R2_active": float(r2_score(_target[_active], _pred[_active])),
            "corr_active": float(np.corrcoef(
                _target[_active], _pred[_active])[0, 1]),
            "SD_active_prediction": float(_pred[_active].std()),
            "RMSE_other_events": float(np.sqrt(np.mean(
                _pred[~_relevant] ** 2))),
            "RMSE_relevant_zero_target": float(np.sqrt(np.mean(
                _pred[_relevant & ~_active] ** 2)))
        }
        _out.append((_metrics, _pred))
        print(_drug, _metrics)
    _best_metrics, _best_pred = max(_out, key=lambda z: z[0]["R2_all"])
    _final_fit = Ridge(alpha=_best_metrics["alpha"]).fit(repair_P, _target)
    repair_decoder_cv[_drug] = {
        "metrics": _best_metrics, "pred": _best_pred,
        "target": _target, "active": _active, "relevant": _relevant
    }
    repair_decoder_fits[_drug] = _final_fit

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Runs 35–38: aligned residual and decoder checks\n"
        "Recovered 4510 antithetic-averaged head features. On the initial "
        "100 training stays, factual observed-minus-predicted slopes against "
        "dose innovations are insulin -1.857 and NE +3.833. These are "
        "small-sample diagnostics, not new causal estimates. Checking "
        "outcome-cache alignment and cross-stay innovation decoding. "
        "No weights edited.\n"
    )
```