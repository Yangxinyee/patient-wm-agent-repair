The tuple check caught a layout error before any edit: column 3 is the dose residual, column 4 is the outcome residual, and column 5 is the predicted dose. I’ll fit the decoders using the corrected columns. Held-out accuracy and non-drug leakage remain the acceptance tests.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
torch.set_num_threads(8)

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
    _rd, _ry = np.asarray(_src[3], float), np.asarray(_src[4], float)
    print(_drug, "residual slope:", np.dot(_rd, _ry)/np.dot(_rd, _rd))
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
    print("active / usable:", int(_active.sum()), int(_ok.sum()),
          "innovation SD:", float(np.std(_vals[_ok])))

repair_post_unique = np.unique(r36_sid)
_perm = np.random.default_rng(712).permutation(repair_post_unique)
repair_post_test = np.isin(r36_sid, _perm[:max(1, len(_perm)//4)])
repair_post_features = {
    "hidden": r36_hidden.detach().numpy().astype(np.float64),
    "input": r36_input.detach().numpy().astype(np.float64),
}
repair_post_decoders, repair_post_results = {}, []

for _feature, _features in repair_post_features.items():
    for _alpha in (1.0, 10.0, 100.0):
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
    _f.write("\n## Runs 42–43: corrected posterior innovation decoding\n"
             "Run 42 assertion caught tuple layout mismatch. Correct layout: "
             "(stay index, exposure hour, dose, dose residual, outcome residual, "
             "predicted dose, predicted outcome). No edits.\n"
             + repr(repair_post_results) + "\n")
```