The saved innovation estimates support lower subsequent glucose after unexpected insulin increases; raw-rate decoding alone is not enough to build the edit. I’ll now test whether the posterior update can decode those innovations while staying quiet on other bundles. Strong held-out decoding would support a targeted latent-state correction; poor decoding would rule it out.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("=== SAVED RESIDUAL TUPLE CHECK ===")
for _drug, _sensor in [("insulin", "glucose_finger"),
                       ("norepinephrine", "map_arterial")]:
    _src = causal_residuals[(_sensor, _drug)]
    print(_drug, [(i, str(a.dtype), float(np.nanmean(a)),
                   float(np.nanstd(a)), np.asarray(a[:3]).tolist())
                  for i, a in enumerate(_src)])
    _b = np.dot(_src[4], _src[5]) / np.dot(_src[4], _src[4])
    print("candidate residual slope:", _b)

repair_post_targets = np.zeros((len(r36_sid), 2), dtype=np.float64)
repair_post_valid = np.ones_like(repair_post_targets, dtype=bool)
repair_post_active = np.asarray(r36_has_start, dtype=bool)
repair_post_panel_si = np.array([r36_sid_to_panel[int(s)] for s in r36_sid])
repair_post_hour = np.floor(r36_hour + 1e-5).astype(int)

for _d, (_drug, _sensor) in enumerate([
    ("insulin", "glucose_finger"), ("norepinephrine", "map_arterial")
]):
    _src = causal_residuals[(_sensor, _drug)]
    _si, _ji = np.asarray(_src[0], int), np.asarray(_src[1], int)
    _rd, _ry = np.asarray(_src[4], float), np.asarray(_src[5], float)
    _beta_check = np.dot(_rd, _ry) / np.dot(_rd, _rd)
    assert np.isfinite(_beta_check)
    assert np.std(_rd) < (3.0 if _d == 0 else 0.5)
    _lookup = np.full((30000, 24), np.nan)
    _lookup[_si, _ji] = _rd
    _active = repair_post_active[:, _d]
    _eligible = _active & (repair_post_hour >= 0) & (repair_post_hour < 24)
    repair_post_valid[_active, _d] = False
    _ix = np.flatnonzero(_eligible)
    _vals = _lookup[repair_post_panel_si[_ix], repair_post_hour[_ix]]
    _ok = np.isfinite(_vals)
    repair_post_targets[_ix[_ok], _d] = _vals[_ok]
    repair_post_valid[_ix[_ok], _d] = True
    print(_drug, "active:", int(_active.sum()),
          "usable active:", int((_active & repair_post_valid[:, _d]).sum()),
          "innovation SD:", float(np.std(_vals[_ok])))

repair_post_unique = np.unique(r36_sid)
_rng = np.random.default_rng(712)
_perm = _rng.permutation(repair_post_unique)
repair_post_test = np.isin(r36_sid, _perm[:max(1, len(_perm)//4)])
repair_post_features = {
    "hidden": r36_hidden.detach().numpy().astype(np.float64),
    "input": r36_input.detach().numpy().astype(np.float64),
}
repair_post_decoders = {}
repair_post_results = []

for _feature, _features in repair_post_features.items():
    for _alpha in (10.0, 100.0):
        for _d, _drug in enumerate(("insulin", "norepinephrine")):
            _valid = repair_post_valid[:, _d]
            _fit = _valid & ~repair_post_test
            _test = _valid & repair_post_test
            _model = Ridge(alpha=_alpha, fit_intercept=True)
            _model.fit(_features[_fit], repair_post_targets[_fit, _d])
            _pred = _model.predict(_features)
            _act_test = _test & repair_post_active[:, _d]
            _quiet_test = _test & ~repair_post_active[:, _d]
            _row = {
                "feature": _feature, "alpha": _alpha, "drug": _drug,
                "active_n": int(_act_test.sum()),
                "active_r2": float(r2_score(
                    repair_post_targets[_act_test, _d], _pred[_act_test])),
                "active_rmse": float(np.sqrt(np.mean(
                    (_pred[_act_test]-repair_post_targets[_act_test, _d])**2))),
                "nonaction_rms": float(np.sqrt(np.mean(_pred[_quiet_test]**2))),
            }
            repair_post_decoders[(_feature, _alpha, _drug)] = _model
            repair_post_results.append(_row)
            print(_row)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 42: posterior innovation decoding\n"
             "Corrected saved residual tuple handling. Stay-separated tests of linear "
             "dose-innovation decoders in posterior input and hidden features; nonaction "
             "targets are zero and missing active labels are excluded. No edits.\n"
             + repr(repair_post_results) + "\n")
```