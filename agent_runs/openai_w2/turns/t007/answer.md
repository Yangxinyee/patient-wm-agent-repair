Lab readings support an insulin effect of about −1.5 to −1.6 per unit over the next hour, without a detectable jump in lab availability. Past-reading placebos are also near zero. The finger estimate appears somewhat inflated by monitoring changes. I’ll now cross-check this effect with history-adjusted regressions and estimate norepinephrine’s effect using cuff readings. Agreement across methods would strengthen the estimates; disagreement would require further investigation before editing.

```python
session["python_runs"] += 1
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

feature_columns = []
feature_names = []
for event in measurement_events:
    feature_columns.append(panel[event].to_numpy())
    feature_names.append("current_" + event)
    for lag in [1, 2, 3]:
        feature_columns.append(
            hist[event].groupby(level="stay_id").shift(lag).to_numpy()
        )
        feature_names.append(f"last_{lag}_" + event)

for drug in drug_events:
    for lag in [1, 2, 3]:
        feature_columns.append(
            panel[drug].groupby(level="stay_id").shift(lag).to_numpy()
        )
        feature_names.append(f"dose_lag_{lag}_" + drug)

initial = panel.loc[panel.index.get_level_values("t") <= 0, measurement_events]
initial = initial.groupby(level="stay_id").last()
initial_rows = initial.reindex(stay_numbers)
for event in measurement_events:
    feature_columns.append(initial_rows[event].to_numpy())
    feature_names.append("initial_" + event)
feature_columns.append(hours.astype(float))
feature_names.append("hour")
X_history = np.column_stack(feature_columns).astype(np.float32)
del initial_rows
print("History feature matrix:", X_history.shape)

print("\n=== LESS POLICY-SENSITIVE CHANNELS BY HOUR ===")
hour_summary = panel.groupby(level="t").agg(
    lab_mean=("glucose_lab", "mean"), cuff_mean=("map_cuff", "mean"),
    insulin_mean=("insulin", "mean"), ne_mean=("norepinephrine", "mean")
)
print(hour_summary.round(3).to_string())

dml_results = []
dml_models = {}
rng = np.random.default_rng(901)

def fit_dml(drug, outcome, label, extra_mask, max_rows=240000):
    y_all = np.asarray(outcome, dtype=float)
    mask = (
        (hours >= 0) & (hours < 24) & np.isfinite(y_all) & extra_mask
    )
    rows = np.flatnonzero(mask)
    if len(rows) > max_rows:
        rows = np.sort(rng.choice(rows, max_rows, replace=False))
    X = X_history[rows]
    y = y_all[rows]
    d = panel[drug].to_numpy()[rows]
    ids = stay_numbers[rows]
    fold = ids % 2
    pred_y = np.zeros(len(rows))
    pred_d = np.zeros(len(rows))
    saved = []
    start = time.time()
    with threadpool_limits(limits=8):
        for k in [0, 1]:
            train_mask = fold != k
            test_mask = fold == k
            settings = dict(
                max_iter=100, learning_rate=.07, max_leaf_nodes=31,
                min_samples_leaf=150, l2_regularization=2,
                early_stopping=False, random_state=17+k
            )
            my = HistGradientBoostingRegressor(**settings).fit(X[train_mask], y[train_mask])
            md = HistGradientBoostingRegressor(**settings).fit(X[train_mask], d[train_mask])
            pred_y[test_mask] = my.predict(X[test_mask])
            pred_d[test_mask] = md.predict(X[test_mask])
            saved.append((my, md))
    ry, rd = y-pred_y, d-pred_d
    denom = rd @ rd
    theta = (rd @ ry) / denom
    score = rd * (ry-theta*rd)
    _, groups = np.unique(ids, return_inverse=True)
    cluster_score = np.bincount(groups, weights=score)
    m = len(cluster_score)
    se = np.sqrt(cluster_score @ cluster_score * m/(m-1)) / denom
    result = {
        "label": label, "drug": drug, "n": len(rows),
        "effect_per_rate_unit": round(float(theta), 4),
        "cluster_se": round(float(se), 4),
        "ci95": [round(float(theta-1.96*se), 4), round(float(theta+1.96*se), 4)],
        "dose_residual_sd": round(float(rd.std()), 4),
        "outcome_residual_sd": round(float(ry.std()), 4),
        "seconds": round(time.time()-start, 1)
    }
    dml_results.append(result)
    dml_models[label] = {"models": saved, "rows": rows, "ry": ry, "rd": rd}
    print(result)

print("\n=== STAY-CROSSFITTED HISTORY-ADJUSTED EFFECTS ===")
fit_dml(
    "insulin", next_obs.glucose_lab, "insulin_next_lab",
    panel.glucose_finger.notna().to_numpy()
)
fit_dml(
    "norepinephrine", next_obs.map_cuff, "NE_next_cuff_both_current_MAP",
    panel[["map_arterial", "map_cuff"]].notna().all(axis=1).to_numpy()
)
fit_dml(
    "norepinephrine", next_obs.map_cuff, "NE_next_cuff_any_current_MAP",
    panel[["map_arterial", "map_cuff"]].notna().any(axis=1).to_numpy()
)

print("\n=== SIMPLE TWO-SENSOR TRANSITION REGRESSIONS ===")
transition_results = []
for drug, a, b, outcome in [
    ("insulin", "glucose_finger", "glucose_lab", next_obs.glucose_lab),
    ("norepinephrine", "map_arterial", "map_cuff", next_obs.map_cuff)
]:
    mask = (
        (hours >= 0) & (hours < 24) &
        panel[[a, b]].notna().all(axis=1).to_numpy() &
        np.isfinite(np.asarray(outcome))
    )
    xa = panel[a].to_numpy()[mask]
    xb = panel[b].to_numpy()[mask]
    X = np.column_stack([
        np.ones(mask.sum()), xa, xb, hours[mask]/24,
        panel[drug].to_numpy()[mask]
    ])
    y = np.asarray(outcome)[mask]
    beta = linalg.lstsq(X, y)[0]
    effect, se = clustered_last_coef(X, y, stay_numbers[mask])
    result = {
        "drug": drug, "n": int(mask.sum()),
        "sensor_coefficients": np.round(beta[1:3], 4).tolist(),
        "sum_sensor_coefficients": round(float(beta[1:3].sum()), 4),
        "drug_coefficient": round(effect, 4), "cluster_se": round(se, 4),
        "intercept": round(float(beta[0]), 4)
    }
    transition_results.append(result)
    print(result)

print("\nCOUNTS:", session)
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 6: independent effect checks\n"
        "Clean threshold RD supports insulin next-hour effect about -1.55/unit "
        "on lab readings. Lab availability is smooth at thresholds; "
        "unadjusted past-outcome placebos are null. Finger availability jumps, "
        "and its effect estimate is more negative. Fitting stay-crossfitted "
        "history-adjusted effects and two-sensor transition models.\n"
        + json.dumps({"DML": dml_results, "transitions": transition_results}, indent=2)
        + "\n"
    )
```