The original model predicts almost no response to any dose edit—less than 0.1 units in each readout. That could be correct, or it could mean the model ignores genuine effects. I’ll estimate next-hour dose associations after controlling for pretreatment histories, separately for each sensor. Consistent estimates across sensors would be more credible than the raw associations.

```python
from sklearn.linear_model import Ridge

def make_history(p):
    hours = p.index.get_level_values("hour").to_numpy(dtype=np.float32)
    features = {"intercept": np.ones(len(p), np.float32), "hour": hours / 24}
    medians = {"glucose_finger": 173., "glucose_lab": 143.,
               "map_arterial": 73., "map_cuff": 81.}
    scales = {"glucose_finger": 71., "glucose_lab": 65.,
              "map_arterial": 24., "map_cuff": 18.}
    for c in MEAS:
        s = p[c]
        last = s.groupby(level=0).ffill().fillna(medians[c])
        z = (last.to_numpy() - medians[c]) / scales[c]
        seen_hour = pd.Series(np.where(s.notna(), hours, np.nan), index=p.index)
        seen_hour = seen_hour.groupby(level=0).ffill().fillna(-3).to_numpy()
        features[c + "_last"] = z
        features[c + "_last_sq"] = z ** 2
        features[c + "_age"] = np.minimum(hours - seen_hour, 12) / 12
        features[c + "_observed"] = s.notna().to_numpy(dtype=np.float32)
        for lag in [1, 2, 3]:
            prev = last.groupby(level=0).shift(lag).fillna(medians[c]).to_numpy()
            features[c + f"_lag{lag}"] = (prev - medians[c]) / scales[c]
        baseline = p[c].unstack("hour")[-1].reindex(
            p.index.get_level_values("stay_id")).fillna(medians[c]).to_numpy()
        features[c + "_baseline"] = (baseline - medians[c]) / scales[c]
    for c, scale in [("insulin", 2.), ("norepinephrine", .5)]:
        for lag in [1, 2, 3]:
            features[c + f"_lag{lag}"] = (
                p[c].groupby(level=0).shift(lag).fillna(0).to_numpy() / scale
            )
    return pd.DataFrame(features, index=p.index).astype("float32")

H = make_history(panel)
H_tune = make_history(panel_tune)
hour_array = panel.index.get_level_values("hour").to_numpy()
stay_array = panel.index.get_level_values("stay_id").to_numpy()
D = panel[DRUGS].to_numpy(dtype=np.float64)

def cluster_slope(res_d, res_y, ids):
    gram = res_d.T @ res_d
    beta = np.linalg.solve(gram, res_d.T @ res_y)
    errors = res_y - res_d @ beta
    score = pd.DataFrame(res_d * errors[:, None])
    score["stay"] = ids
    score = score.groupby("stay").sum().to_numpy()
    inv = np.linalg.inv(gram)
    cov = inv @ (score.T @ score) @ inv
    return beta, np.sqrt(np.diag(cov))

linear_results = {}
print("=== HISTORY-ADJUSTED NEXT-HOUR ASSOCIATIONS ===")
for outcome in MEAS:
    y = panel[outcome].groupby(level=0).shift(-1).to_numpy()
    valid = np.isfinite(y) & (hour_array >= 0) & (hour_array <= 23)
    x = H.to_numpy()[valid]
    targets = np.column_stack([y[valid], D[valid]])
    nuisance = Ridge(alpha=10., fit_intercept=False, solver="cholesky")
    nuisance.fit(x, targets)
    resid = targets - nuisance.predict(x)
    beta, se = cluster_slope(resid[:, 1:], resid[:, 0], stay_array[valid])
    linear_results[outcome] = {"beta": beta, "se": se}
    print(outcome, "n", valid.sum(),
          "insulin beta/se", np.round([beta[0], se[0]], 4),
          "NE beta/se", np.round([beta[1], se[1]], 4),
          "dose residual sd", np.round(resid[:, 1:].std(axis=0), 4))

print("\n=== EXAMPLE INSULIN-TREATED STAY ===")
s = panel.insulin.groupby(level=0).sum()
example_id = s.index[s.gt(15)][0]
print(panel.loc[example_id, MEAS + DRUGS].to_string())

print("\n=== EXAMPLE NOREPINEPHRINE-TREATED STAY ===")
s = panel.norepinephrine.groupby(level=0).sum()
example_id = s.index[s.gt(7)][0]
print(panel.loc[example_id, MEAS + DRUGS].to_string())

Path("/work/notes.md").open("a").write(
    "\n## Forward 1 and run 4\n"
    "- Original rollout on first 100 tuning stays: insulin x2 +0.026, half -0.085 mg/dL; "
    "NE x2 -0.098, half +0.018 mmHg. These are predictions, not established truth.\n"
    "- Constructed pretreatment histories: latest sensors, ages, availability, three lags, "
    "baseline readings and past doses. Initial linear residualization is diagnostic, not sufficient identification.\n"
    "- Python runs: 4. Forward passes: 1.\n"
)
```