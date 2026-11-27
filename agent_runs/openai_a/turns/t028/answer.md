The insulin discontinuity estimates are consistent: about −1.9 mg/dL per added unit/hour, with near-zero pre-treatment placebo effects. The later-dose-adjusted impulse fit suggests persistence around 0.83, implying a larger sustained effect than the model’s −1.85 rollout. I’ll calculate that sustained target and separately test norepinephrine; I won’t assume its effect is positive or absent.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

torch.set_num_threads(8)
threadpool_limits(limits=8)

def causal_shift_table(panel_array, ids, outcome, drug, beta, rho):
    di = columns.index(drug)
    yi = columns.index(outcome)
    dose = panel_array[:, :, di]
    exposure = np.zeros_like(dose, dtype=np.float64)
    reached = np.zeros_like(dose, dtype=bool)
    active = np.zeros(len(ids), dtype=bool)
    carry = np.zeros(len(ids), dtype=np.float64)
    for t in range(n_times):
        if t:
            previous_hour = hour_grid[t - 1]
            increment = dose[:, t - 1] if previous_hour >= 12 else 0.
            carry = rho * carry + beta * increment
            if previous_hour >= 12:
                active |= dose[:, t - 1] > 0
        exposure[:, t] = carry
        reached[:, t] = active
    mask = (np.isfinite(panel_array[:, :, yi]) & reached &
            (hour_grid[None, :] >= 13) & (hour_grid[None, :] <= 24))
    return float(exposure[mask].mean()), int(mask.sum()), exposure, reached

ins_target = causal_shift_table(P, train_ids, "glucose_finger", "insulin",
                               rd_dynamic_fit.x[0], rd_dynamic_fit.x[1])
TP = tpanel.to_numpy(dtype=np.float32).reshape(-1, n_times, n_cols)
tuning_ids = tpanel.index.get_level_values("stay_id").unique().to_numpy()
targ_tuning = causal_shift_table(TP[:100], tuning_ids[:100],
                                "glucose_finger", "insulin",
                                rd_dynamic_fit.x[0], rd_dynamic_fit.x[1])
print("Record-derived insulin x2 targets, train / first 100 tuning:",
      ins_target[:2], targ_tuning[:2])
print("Insulin half targets:", -.5 * ins_target[0], -.5 * targ_tuning[0])
print("Dose output example:")
print(dose_baseline.head(8).to_string(index=False))

print("\n=== FOUR NOREPINEPHRINE TRAJECTORIES ===")
ne_array = P[:, :, columns.index("norepinephrine")]
examples = np.flatnonzero(ne_array[:, 13:].sum(axis=1) > 2)[:4]
for i in examples:
    print(train_ids[i])
    print(pd.DataFrame({"hour": hour_grid, "arterial": P[i, :, 3],
                        "cuff": P[i, :, 4], "ne": P[i, :, 5]}).to_string(index=False))

def make_ne_data(n_lags):
    features = []
    names = []
    for code in ("map_arterial", "map_cuff", "norepinephrine"):
        for lag in range(0 if code != "norepinephrine" else 1, n_lags + 1):
            val = shifted_col(code, -lag).astype(np.float32)
            features.append(val)
            names.append(f"{code}_lag{lag}")
    for code in ("map_arterial", "map_cuff"):
        c = columns.index(code)
        for ti in (0, 1):
            features.append(np.repeat(P[:, ti, c], n_times))
            names.append(f"{code}_admission{ti}")
    features.append(np.tile(hour_grid, n_stays).astype(np.float32))
    names.append("hour")
    X = np.column_stack(features)
    y = shifted_col("map_arterial", 1).astype(np.float32)
    dose = P[:, :, 5].reshape(-1)
    keep = (time_ix >= n_lags + 1) & (time_ix < n_times - 1) & np.isfinite(y) & np.isfinite(P[:, :, 3].reshape(-1))
    return X[keep], y[keep], dose[keep], stay_ix[keep], names

ne_dml_results = {}
ne_dml_cache = {}
for n_lags in (1, 4, 8):
    start = time.time()
    X, y, dose, groups, fnames = make_ne_data(n_lags)
    dr = np.zeros(len(y), dtype=np.float64)
    yr = np.zeros(len(y), dtype=np.float64)
    for fold in (0, 1):
        fit = (groups % 2 != fold)
        test = ~fit
        for target, residual in ((dose, dr), (y, yr)):
            estimator = HistGradientBoostingRegressor(
                max_iter=120, max_leaf_nodes=23, min_samples_leaf=100,
                learning_rate=.08, l2_regularization=20, random_state=123)
            estimator.fit(X[fit], target[fit])
            residual[test] = target[test] - estimator.predict(X[test])
    results = {}
    for label, mask in [("all", np.ones(len(y), bool)),
                        ("interior", (dose > .02) & (dose < .98))]:
        a, b, gr = dr[mask], yr[mask], groups[mask]
        beta = float(a @ b / (a @ a))
        scores = np.bincount(gr, weights=a * (b - beta * a), minlength=n_stays)
        se = float(np.linalg.norm(scores) / (a @ a))
        results[label] = {"beta": beta, "se": se, "n": int(mask.sum()),
                          "dose_residual_sd": float(np.std(a)),
                          "outcome_residual_sd": float(np.std(b))}
    ne_dml_results[n_lags] = results
    ne_dml_cache[n_lags] = {"dose_residual": dr, "outcome_residual": yr, "groups": groups}
    print("NE DML", n_lags, json.dumps(results), "seconds", round(time.time() - start, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Sustained insulin target and NE flexible adjustment; Python 24, forward 4\n")
    f.write("- RD beta/persistence: " + str(rd_dynamic_fit.x.tolist()) + ".\n")
    f.write("- Fixed-record insulin x2 target, train / first 100 tuning: " +
            str((ins_target[:2], targ_tuning[:2])) + "; half is -0.5 times x2.\n")
    f.write("- Flexible, stay-held-out NE estimates remain observational, not automatically causal.\n")
    f.write(json.dumps(ne_dml_results, indent=2) + "\n")
```
