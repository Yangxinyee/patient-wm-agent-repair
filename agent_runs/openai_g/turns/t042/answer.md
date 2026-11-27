The existing histories use only information available before each infusion. I’ll use them to estimate the association between a late norepinephrine dose and average MAP over the following 12 hours, with folds separated by stay. A stable nonzero window-average would challenge the null hypothesis, though this observational check is not the full multi-dose intervention.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from pathlib import Path
import time

t_window_start = time.time()
window_sum_ne = np.zeros(720000, dtype=np.float64)
window_count_ne = np.zeros(720000, dtype=np.int16)
for window_lag in range(1, 13):
    yf = np.asarray(future_reading_v2(2, window_lag), dtype=np.float64)
    assert yf.shape == (720000,)
    ok = np.isfinite(yf)
    window_sum_ne[ok] += yf[ok]
    window_count_ne[ok] += 1

window_mean_ne = np.full(720000, np.nan)
np.divide(window_sum_ne, window_count_ne, out=window_mean_ne,
          where=window_count_ne > 0)
window_change_ne = window_mean_ne - map_baseline_v3
window_eligible_ne = (
    positive_ne_v2 & (hours_all >= 4) & (hours_all <= 12)
    & np.isfinite(window_change_ne) & np.isfinite(A_ne_v2)
)
assert np.all(row_folds_v2.reshape(30000, 24)
              == row_folds_v2.reshape(30000, 24)[:, :1])
assert set(np.unique(row_folds_v2)) == {0, 1, 2}

window_pa_ne = np.full(720000, np.nan)
window_py_ne = np.full(720000, np.nan)
print("Eligible rows:", int(window_eligible_ne.sum()),
      "stays:", len(np.unique(groups_all[window_eligible_ne])))

with threadpool_limits(limits=8):
    for wf in range(3):
        tr = window_eligible_ne & (row_folds_v2 != wf)
        te = window_eligible_ne & (row_folds_v2 == wf)
        print("Fold", wf, "train", int(tr.sum()), "test", int(te.sum()), flush=True)
        dose_fit_window = HistGradientBoostingRegressor(**settings_v3)
        dose_fit_window.fit(X_ne_v2[tr], A_ne_v2[tr])
        window_pa_ne[te] = dose_fit_window.predict(X_ne_v2[te])
        outcome_fit_window = HistGradientBoostingRegressor(**settings_v3)
        outcome_fit_window.fit(X_ne_v2[tr], window_change_ne[tr])
        window_py_ne[te] = outcome_fit_window.predict(X_ne_v2[te])

window_ra_ne = A_ne_v2 - window_pa_ne
window_ry_ne = window_change_ne - window_py_ne
window_results_ne = []
window_selections_ne = {
    "origins_4_to_12": window_eligible_ne,
    "origin_12": window_eligible_ne & (hours_all == 12),
    "origins_4_to_11": window_eligible_ne & (hours_all < 12),
    "origin_12_below_max_rate": (
        window_eligible_ne & (hours_all == 12)
        & (A_ne_v2 < np.nanmax(A_ne_v2))
    )
}
for wl, sel in window_selections_ne.items():
    rr = orthogonal_coefficient_v2(window_ra_ne, window_ry_ne, sel)
    rr.update(subset=wl, target="mean_MAP_next_12_hours_minus_pre_dose_baseline")
    window_results_ne.append(rr)
    print(rr)

print("Dose prediction residual SD:",
      float(np.nanstd(window_ra_ne[window_eligible_ne])))
print("Future arterial readings per origin:",
      np.quantile(window_count_ne[window_eligible_ne], [0, .25, .5, .75, 1]))
print("Elapsed seconds:", round(time.time() - t_window_start, 1))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## NE 12-hour window-average diagnostic, run 34\n")
    f.write("500-tree, three-stay-fold nuisance fits on matching positive-dose "
            "rows at origins 4..12, adjusting only pre-dose history. "
            "Target averages recorded arterial MAP over the next 12 hours "
            "and subtracts a pre-dose baseline. This is a single-dose "
            "observational diagnostic, not an identified multi-dose rollout "
            "effect or an independent replication. No weights changed.\n")
    f.write(str(window_results_ne) + "\n")
```