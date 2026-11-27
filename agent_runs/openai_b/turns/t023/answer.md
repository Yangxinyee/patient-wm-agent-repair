The quadratic checks found no consistent response, but one norepinephrine four-hour test was marginally significant. I won’t treat that as settled. I’ll repeat the analysis with richer history models and add a pre-infusion placebo outcome. A stable future-only signal would challenge the edit; disappearing signals or comparable placebo associations would suggest residual bias.

```python
sensitive_params = dict(check_params)
sensitive_params.update(
    max_iter=300, max_leaf_nodes=31, min_samples_leaf=80,
    learning_rate=0.05, l2_regularization=1.0, early_stopping=False
)
print("Previous nuisance parameters:", check_params)
print("Sensitivity parameters:", sensitive_params)
sensitivity_checks = []
started = time.time()

def fit_quadratic_moment(XX, target, dose, ids, folds, model_params):
    targets = np.column_stack([target, dose, dose ** 2])
    predictions = np.empty_like(targets)
    for fold in np.unique(folds):
        test = folds == fold
        for column in range(3):
            nuisance = HistGradientBoostingRegressor(**model_params)
            nuisance.fit(
                XX.iloc[np.flatnonzero(~test)], targets[~test, column]
            )
            predictions[test, column] = nuisance.predict(
                XX.iloc[np.flatnonzero(test)]
            )
    residual = targets - predictions
    y = residual[:, 0] - residual[:, 0].mean()
    R = residual[:, 1:] - residual[:, 1:].mean(axis=0)
    inverse = np.linalg.pinv(R.T @ R)
    beta = inverse @ (R.T @ y)
    score = pd.DataFrame(
        R * (y - R @ beta)[:, None], columns=["linear", "quadratic"]
    ).groupby(ids, sort=False).sum().to_numpy()
    groups = len(score)
    cov = inverse @ (score.T @ score) @ inverse * groups / (groups - 1)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    wald = float(beta @ np.linalg.pinv(cov) @ beta)
    return {
        "n": len(y),
        "linear_beta": float(beta[0]), "linear_se": float(se[0]),
        "quadratic_beta": float(beta[1]), "quadratic_se": float(se[1]),
        "joint_p": float(chi2.sf(wald, 2)),
    }

with threadpool_limits(limits=8):
    for drug, outcome, companion, other, scale in [
        ("insulin", "glucose_finger", "glucose_lab", "norepinephrine", 1.0),
        ("norepinephrine", "map_arterial", "map_cuff", "insulin", 0.1),
    ]:
        rate = panel[drug].fillna(0).to_numpy(dtype=float)
        now = panel[outcome].to_numpy(dtype=float)
        context = Xhistory.copy()
        context[other + "_current"] = panel[other].fillna(0)
        running = (
            (rate > 0)
            & (Xhistory[drug + "_lag1"].to_numpy() > 0)
            & np.isfinite(now)
            & (analysis_hours >= 12)
        )
        both = panel[companion].notna().to_numpy()
        past = panel[outcome].groupby(
            analysis_stays, sort=False
        ).shift(1).to_numpy(dtype=float)

        for horizon in (0, 1, 4):
            if horizon == 0:
                target_full = now - past
                possible = np.isfinite(past) & (analysis_hours <= 23)
                strata = {"both_sensors_running": both}
            else:
                future = panel[outcome].groupby(
                    analysis_stays, sort=False
                ).shift(-horizon).to_numpy(dtype=float)
                target_full = future - now
                possible = (
                    np.isfinite(future)
                    & (analysis_hours <= 24 - horizon)
                )
                strata = {
                    "all_running": np.ones(len(panel), dtype=bool),
                    "both_sensors_running": both,
                }

            for label, selection in strata.items():
                idx = np.flatnonzero(running & possible & selection)
                result = fit_quadratic_moment(
                    context.iloc[idx], target_full[idx], rate[idx] / scale,
                    np.asarray(analysis_stays)[idx],
                    np.asarray(analysis_fold)[idx], sensitive_params
                )
                result.update(
                    drug=drug, horizon=horizon, stratum=label,
                    rate_scale=scale,
                    outcome_type=("pre_infusion_placebo" if horizon == 0
                                  else "future_change")
                )
                sensitivity_checks.append(result)

print("\n=== NUISANCE SENSITIVITY AND PLACEBOS ===")
print(pd.DataFrame(sensitivity_checks).round(5).to_string(index=False))
print("Elapsed seconds:", round(time.time() - started, 1))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 15: investigate marginal nonlinear signal\n"
        "Run 14 NE horizon-4 quadratic joint p=0.046 was isolated among "
        "four checks; insulin joint p=0.124/0.575 and NE horizon-1 "
        "p=0.511. Testing richer nuisance models, all-running versus "
        "both-sensor subsets, and current-minus-previous reading as a "
        "pre-infusion placebo. Significant placebo moments diagnose "
        "estimation bias; null placebos cannot prove unconfoundedness.\n"
        "Parameters: " + json.dumps(sensitive_params) + "\n"
        + json.dumps(sensitivity_checks) + "\n"
    )
```