The factual baseline is 1.50162 nats per reading on this cohort. Before making the model dose-invariant, I’ll check outcomes up to 12 hours later and repeat the one-hour estimates with stronger history models. Detectable delayed effects, or estimates that change materially with better controls, would refute the null-effect hypothesis.

```python
start = time.time()
delayed_results = {}
print("=== HELD-OUT DELAYED EFFECT DIAGNOSTICS ===")
for horizon in [2, 3, 6, 12]:
    for outcome, j in [
        ("glucose_finger", 0), ("glucose_lab", 0),
        ("map_arterial", 1), ("map_cuff", 1)
    ]:
        y = panel[outcome].groupby(level=0).shift(-horizon).to_numpy()
        fit = fit_rows & np.isfinite(y) & (hour_array <= 24 - horizon)
        test = test_rows & np.isfinite(y) & (hour_array <= 24 - horizon)
        m = HistGradientBoostingRegressor(**params)
        m.fit(Xrich[fit], y[fit])
        ry = y[test] - m.predict(Xrich[test])
        rd = D[test] - dose_pred[test]
        beta, se = cluster_slope(rd, ry, stay_array[test])
        delayed_results[(horizon, outcome)] = (beta, se, int(test.sum()))
        print("horizon", horizon, outcome, "n", test.sum(),
              "target-drug beta/se", np.round([beta[j], se[j]], 4))

print("\n=== STRONGER ONE-HOUR NUISANCE MODELS ===")
strong_params = dict(params)
strong_params.update(max_iter=350, max_leaf_nodes=63, min_samples_leaf=100)
dose_pred_strong = np.full_like(dose_pred, np.nan)
dose_models_strong = {}
for j, drug in enumerate(DRUGS):
    m = HistGradientBoostingRegressor(**strong_params)
    m.fit(Xrich[fit_rows], D[fit_rows, j])
    dose_pred_strong[test_rows, j] = m.predict(Xrich[test_rows])
    dose_models_strong[drug] = m

strong_results = {}
for outcome, j in [
    ("glucose_finger", 0), ("glucose_lab", 0),
    ("map_arterial", 1), ("map_cuff", 1)
]:
    y = panel[outcome].groupby(level=0).shift(-1).to_numpy()
    fit = fit_rows & np.isfinite(y)
    test = test_rows & np.isfinite(y)
    m = HistGradientBoostingRegressor(**strong_params)
    m.fit(Xrich[fit], y[fit])
    ry = y[test] - m.predict(Xrich[test])
    rd = D[test] - dose_pred_strong[test]
    beta, se = cluster_slope(rd, ry, stay_array[test])
    strong_results[outcome] = (beta, se, int(test.sum()))
    print(outcome, "target-drug beta/se", np.round([beta[j], se[j]], 4),
          "dose residual sd", np.round(rd[:, j].std(), 4))

print("elapsed seconds:", round(time.time() - start, 1))
Path("/work/notes.md").open("a").write(
    "\n## Run 6: challenge the null-effect hypothesis\n"
    "- Nonlinear held-out one-hour effects: insulin/finger -0.0031 (SE .0845), "
    "insulin/lab -0.0557 (.1296), NE/arterial +0.1263 (.1810), NE/cuff -0.4367 (.3711).\n"
    "- Checked local projections at horizons 2, 3, 6, 12 and stronger nuisance regressions.\n"
    "- Original factual baseline first 100 tuning stays: 1.50162 nats per value, 4,787 values.\n"
    "- Current hypothesis: both drugs have negligible effects in this simulated ICU; "
    "this is based on records, not clinical expectations.\n"
    "- Python runs: 6. Forward passes: 2.\n"
)
```