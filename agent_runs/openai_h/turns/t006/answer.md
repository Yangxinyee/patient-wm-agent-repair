The linear estimates differ substantially between sensors, so I don’t trust them as causal effects yet. I’ll use richer, nonlinear history controls and evaluate residual dose–outcome relationships on held-out stays. Estimates that stabilize across sensors and later ICU hours would support a real effect; estimates that collapse toward zero would support the original model.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

thread_limits = threadpool_limits(limits=8)

def enrich_history(p, base):
    out = base.copy()
    for c, median, scale in [
        ("glucose_finger", 173., 71.), ("glucose_lab", 143., 65.),
        ("map_arterial", 73., 24.), ("map_cuff", 81., 18.)
    ]:
        raw = p[c]
        last = raw.groupby(level=0).ffill().fillna(median)
        for lag in [4, 5, 6]:
            out[c + f"_lag{lag}"] = (
                last.groupby(level=0).shift(lag).fillna(median) - median
            ) / scale
        count = raw.notna().groupby(level=0).cumsum()
        total = raw.fillna(0).groupby(level=0).cumsum()
        out[c + "_history_mean"] = (total.div(count.replace(0, np.nan)).fillna(median) - median) / scale
        out[c + "_raw_current"] = (raw - median) / scale

    fused = {
        "G": p[["glucose_finger", "glucose_lab"]].mean(axis=1),
        "M": pd.concat([p.map_arterial, p.map_cuff - 3.], axis=1).mean(axis=1)
    }
    for c, raw in fused.items():
        median, scale = (165., 70.) if c == "G" else (75., 22.)
        last = raw.groupby(level=0).ffill().fillna(median)
        for lag in range(7):
            s = last if lag == 0 else last.groupby(level=0).shift(lag).fillna(median)
            out[c + f"_lag{lag}"] = (s - median) / scale
        count = raw.notna().groupby(level=0).cumsum()
        total = raw.fillna(0).groupby(level=0).cumsum()
        out[c + "_history_mean"] = (total.div(count.replace(0, np.nan)).fillna(median) - median) / scale

    for c, scale in [("insulin", 2.), ("norepinephrine", .5)]:
        for lag in [4, 5, 6]:
            out[c + f"_lag{lag}"] = p[c].groupby(level=0).shift(lag).fillna(0) / scale
    return out.astype("float32")

Hrich = enrich_history(panel, H)
Hrich_tune = enrich_history(panel_tune, H_tune)
Xrich = Hrich.to_numpy()
eligible = (hour_array >= 0) & (hour_array <= 23)
held = (stay_array % 5 == 0)
fit_rows = eligible & ~held
test_rows = eligible & held

params = dict(
    max_iter=150, learning_rate=.08, max_leaf_nodes=31,
    min_samples_leaf=200, l2_regularization=20.,
    early_stopping=False, random_state=1701
)
dose_models, outcome_models = {}, {}
dose_pred = np.full((len(panel), 2), np.nan, dtype=np.float32)
outcome_pred = {}
start = time.time()

print("history features:", Xrich.shape[1], "fit rows:", fit_rows.sum(),
      "held-out rows:", test_rows.sum())
for j, drug in enumerate(DRUGS):
    m = HistGradientBoostingRegressor(**params)
    m.fit(Xrich[fit_rows], D[fit_rows, j])
    dose_pred[test_rows, j] = m.predict(Xrich[test_rows])
    dose_models[drug] = m
    print("dose nuisance", drug, "elapsed", round(time.time() - start, 1))

dml_results = {}
for outcome in MEAS:
    y = panel[outcome].groupby(level=0).shift(-1).to_numpy()
    fit = fit_rows & np.isfinite(y)
    test = test_rows & np.isfinite(y)
    m = HistGradientBoostingRegressor(**params)
    m.fit(Xrich[fit], y[fit])
    pred = np.full(len(panel), np.nan, np.float32)
    pred[test] = m.predict(Xrich[test])
    outcome_models[outcome] = m
    outcome_pred[outcome] = pred
    rd = D[test] - dose_pred[test]
    ry = y[test] - pred[test]
    ids = stay_array[test]
    hh = hour_array[test]
    results = {}
    print("\nOUTCOME", outcome, "elapsed", round(time.time() - start, 1))
    for label, sub in [
        ("all", np.ones(len(ry), dtype=bool)),
        ("hours_8_to_23", hh >= 8),
        ("same_sensor_at_t", panel[outcome].notna().to_numpy()[test]),
        ("both_target_sensors_at_t",
         panel[MEAS[:2] if outcome.startswith("glucose") else MEAS[2:]].notna().all(axis=1).to_numpy()[test])
    ]:
        if sub.sum() < 1000:
            continue
        beta, se = cluster_slope(rd[sub], ry[sub], ids[sub])
        results[label] = (beta, se, int(sub.sum()))
        print(label, "n", sub.sum(),
              "insulin beta/se", np.round([beta[0], se[0]], 4),
              "NE beta/se", np.round([beta[1], se[1]], 4),
              "dose residual sd", np.round(rd[sub].std(axis=0), 4))
    dml_results[outcome] = results

Path("/work/notes.md").open("a").write(
    "\n## Run 5: nonlinear held-out residualization\n"
    "- Initial linear associations were sensor-dependent: insulin/glucose +1.46 to +2.44 per unit; "
    "NE/MAP -2.17 to -7.27 per rate unit. Not accepted as causal.\n"
    "- Richer histories include six lags, cumulative sensor means, raw current readings, and fused sensors.\n"
    "- Nuisance regressions fit on 80% of training stays; residual effects evaluated on disjoint 20%.\n"
    "- Python runs: 5. Forward passes: 1.\n"
)
```