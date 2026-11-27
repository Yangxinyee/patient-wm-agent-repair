No new causal-fit result has arrived. I’ll run the decisive comparisons first: lags 4, 10, and 12, with dose and MAP-change models fitted on the same late-hour sample. Agreement across these lags would strengthen evidence of an effect; another isolated lag-12 estimate would not.

```python
import time, json
from sklearn.ensemble import HistGradientBoostingRegressor

map_baseline_v3 = pd.DataFrame(
    np.where(np.isfinite(arr[:, :, 2]), arr[:, :, 2], arr[:, :, 3])
).ffill(axis=1).to_numpy()[:, 1:25].reshape(-1)
settings_v3 = dict(max_iter=500, max_leaf_nodes=31, min_samples_leaf=150,
                   max_bins=128, learning_rate=.05, l2_regularization=3.,
                   early_stopping=False, random_state=2301)
ne_late_checks_v3, ne_late_residuals_v3 = {}, {}

def matched_crossfit_v3(X, exposure, target, eligible, label):
    eligible = eligible & np.isfinite(exposure) & np.isfinite(target)
    pa3, py3 = np.full(len(target), np.nan), np.full(len(target), np.nan)
    for fold3 in range(3):
        tr3, te3 = eligible & (row_folds_v2 != fold3), eligible & (row_folds_v2 == fold3)
        for response3, output3 in [(exposure, pa3), (target, py3)]:
            fit3 = HistGradientBoostingRegressor(**settings_v3)
            fit3.fit(X[tr3], response3[tr3])
            output3[te3] = fit3.predict(X[te3])
    print(label, "n", int(eligible.sum()), flush=True)
    return exposure - pa3, target - py3

_t3 = time.time()
for _h3 in [4, 10, 12]:
    _ra3, _ry3 = matched_crossfit_v3(
        X_ne_v2, A_ne_v2, future_reading_v2(2, _h3) - map_baseline_v3,
        positive_ne_v2 & (treatment_hours_v2 >= 4), "NE lag " + str(_h3))
    ne_late_residuals_v3[_h3] = (_ra3, _ry3)
    for _cut3 in [4, 12]:
        _r3 = orthogonal_coefficient_v2(_ra3, _ry3, treatment_hours_v2 >= _cut3)
        _r3.update(horizon=_h3, hour_min=_cut3, adjustment="late_only_matched_change")
        ne_late_checks_v3[(_h3, _cut3)] = _r3
        print(json.dumps(_r3), flush=True)
print("Elapsed seconds:", round(time.time() - _t3, 1))
with open("/work/notes.md", "a") as _f:
    _f.write("\n\n## NE late-only matched MAP-change checks\n"
             "500-tree nuisance fits, cross-fitted by stay, trained on identical dose/outcome "
             "samples at hours >=4. No weight changes.\n"
             + json.dumps(list(ne_late_checks_v3.values())) + "\n")
```