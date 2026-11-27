Across 200 tuning stays, the NLL change is essentially zero, with a 95% interval far below the allowed increase. I’ll now estimate insulin effects using stay-level cross-fitting and strictly pre-dose histories, emphasizing hour 12 onward. A consistent late-hour effect would challenge the zero-effect edit; small, unstable estimates would support retaining it.

```python
import time, json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

threadpool_limits(limits=8)
analysis_started = time.time()
n_stays, n_times, _ = arr.shape
treatment_hours_v2 = np.tile(np.arange(24), n_stays)
stay_groups_v2 = np.repeat(np.arange(n_stays), 24)
rng_v2 = np.random.default_rng(71031)
stay_folds_v2 = rng_v2.permutation(n_stays) % 3
row_folds_v2 = np.repeat(stay_folds_v2, 24)

def prefix_last_v2(v):
    ix = np.arange(v.shape[1])[None, :]
    last_ix = np.maximum.accumulate(np.where(np.isfinite(v), ix, -1), axis=1)
    last = np.take_along_axis(v, np.maximum(last_ix, 0), axis=1).copy()
    last[last_ix < 0] = np.nan
    age = (ix - last_ix).astype(np.float32)
    age[last_ix < 0] = np.nan
    return last, age

def pre_dose_features_v2(drug):
    columns, names = [], []
    def put(name, values):
        names.append(name)
        columns.append(np.asarray(values[:, 1:25], dtype=np.float32).reshape(-1))
    put("hour", np.broadcast_to(np.arange(-1, 25), (n_stays, 26)))
    own = [0, 1] if drug == "insulin" else [2, 3]
    for c in range(4):
        v = arr[:, :, c]
        last, age = prefix_last_v2(v)
        put(events[c] + "_current", v)
        put(events[c] + "_last", last)
        put(events[c] + "_age", age)
        if c not in own:
            continue
        ix = np.arange(26)[None, :]
        first_ix = np.minimum.accumulate(np.where(np.isfinite(v), ix, 26), axis=1)
        first = np.take_along_axis(v, np.minimum(first_ix, 25), axis=1).copy()
        first[first_ix == 26] = np.nan
        put(events[c] + "_first_so_far", first)
        for lag in range(1, 9):
            shifted = np.full_like(v, np.nan)
            shifted[:, lag:] = v[:, :-lag]
            put(events[c] + f"_lag{lag}", shifted)
        sums = np.concatenate([np.zeros((n_stays, 1), np.float32),
                               np.cumsum(np.nan_to_num(v), axis=1)], axis=1)
        counts = np.concatenate([np.zeros((n_stays, 1), np.int32),
                                 np.cumsum(np.isfinite(v), axis=1)], axis=1)
        for width in (3, 6, 12):
            ends = np.arange(1, 27)
            starts = np.maximum(ends - width, 0)
            num = sums[:, ends] - sums[:, starts]
            den = counts[:, ends] - counts[:, starts]
            mean = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            put(events[c] + f"_mean{width}", mean)
            put(events[c] + f"_count{width}", den)
    for c in (4, 5):
        for lag in range(1, 13):
            shifted = np.full_like(arr[:, :, c], np.nan)
            shifted[:, lag:] = arr[:, :-lag, c]
            put(events[c] + f"_rate_lag{lag}", shifted)
    return np.column_stack(columns), names

nuisance_settings_v2 = dict(max_iter=180, max_leaf_nodes=31,
                            min_samples_leaf=200, max_bins=128,
                            learning_rate=0.07, l2_regularization=2.0,
                            early_stopping=False, random_state=8107)

def crossfit_mean_v2(X, target, eligible, label):
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    eligible = eligible & np.isfinite(target)
    for fold in range(3):
        train_mask = eligible & (row_folds_v2 != fold)
        test_mask = eligible & (row_folds_v2 == fold)
        model = HistGradientBoostingRegressor(**nuisance_settings_v2)
        model.fit(X[train_mask], target[train_mask])
        prediction[test_mask] = model.predict(X[test_mask])
        print(label, "fold", fold, "train", int(train_mask.sum()),
              "test", int(test_mask.sum()), "elapsed", round(time.time() - analysis_started, 1),
              flush=True)
    return prediction

def future_reading_v2(channel, horizon):
    target = np.full((n_stays, 24), np.nan, dtype=np.float32)
    target[:, :25-horizon] = arr[:, 1+horizon:26, channel]
    return target.reshape(-1)

def orthogonal_coefficient_v2(a_res, y_res, selection):
    use = selection & np.isfinite(a_res) & np.isfinite(y_res)
    a, y, groups = a_res[use], y_res[use], stay_groups_v2[use]
    denominator = float(a @ a)
    beta = float(a @ y / denominator)
    scores = np.bincount(groups, weights=a * (y - beta * a), minlength=n_stays)
    G = len(np.unique(groups))
    se = float(np.sqrt(G / (G-1) * (scores @ scores)) / denominator)
    return {"n": int(use.sum()), "stays": G, "coef": beta, "se": se,
            "ci95": [beta - 1.96*se, beta + 1.96*se],
            "dose_res_sd": float(np.std(a))}

X_insulin_v2, insulin_features_v2 = pre_dose_features_v2("insulin")
A_insulin_v2 = arr[:, 1:25, 4].reshape(-1).astype(np.float64)
positive_insulin_v2 = np.isfinite(A_insulin_v2) & (A_insulin_v2 > 0)
pA_insulin_v2 = crossfit_mean_v2(X_insulin_v2, A_insulin_v2,
                               positive_insulin_v2, "insulin dose")
rA_insulin_v2 = A_insulin_v2 - pA_insulin_v2
insulin_results_v2 = []
insulin_y_residuals_v2 = {}
for horizon in (1, 4):
    Y = future_reading_v2(0, horizon)
    pY = crossfit_mean_v2(X_insulin_v2, Y, positive_insulin_v2,
                         f"glucose horizon {horizon}")
    rY = Y - pY
    insulin_y_residuals_v2[horizon] = rY
    selections = {
        "all_positive": positive_insulin_v2,
        "hour_ge12": positive_insulin_v2 & (treatment_hours_v2 >= 12),
        "hour_ge12_current_observed": positive_insulin_v2 & (treatment_hours_v2 >= 12)
            & np.isfinite(arr[:, 1:25, 0].reshape(-1)),
        "hour_ge12_rate_lt12": positive_insulin_v2 & (treatment_hours_v2 >= 12)
            & (A_insulin_v2 < 12)
    }
    for label, selection in selections.items():
        result = {"drug": "insulin", "outcome": "glucose_finger",
                  "horizon": horizon, "subset": label,
                  **orthogonal_coefficient_v2(rA_insulin_v2, rY, selection)}
        insulin_results_v2.append(result)
        print(json.dumps(result), flush=True)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Strictly pre-dose insulin checks\n"
            "Three folds separated by stay; nuisance regressions fit on positive rates. "
            "All first-value features are prefix-only (no future baseline leakage). "
            "No post-dose variables used as adjustment covariates.\n"
            + json.dumps(insulin_results_v2) + "\n")
print("Completed in", round(time.time() - analysis_started, 1), "seconds")
```
