Across all 400 tuning stays, NLL improved by 0.000068 nats over 18,910 readings. Late-period effects remain near zero overall, although norepinephrine’s subgroup estimates vary in sign. I’ll test curved dose responses among continuing infusions with both sensors observed. A stable nonlinear signal across horizons would refute dose invariance.

```python
from scipy.stats import chi2

quadratic_checks = []
started = time.time()

with threadpool_limits(limits=8):
    for drug, outcome, companion, other, scale in [
        ("insulin", "glucose_finger", "glucose_lab", "norepinephrine", 1.0),
        ("norepinephrine", "map_arterial", "map_cuff", "insulin", 0.1),
    ]:
        actual_rate = panel[drug].fillna(0).to_numpy(dtype=float)
        scaled_rate = actual_rate / scale
        now = panel[outcome].to_numpy(dtype=float)
        context = Xhistory.copy()
        context[other + "_current"] = panel[other].fillna(0)
        complete_running = (
            (actual_rate > 0)
            & (Xhistory[drug + "_lag1"].to_numpy() > 0)
            & np.isfinite(now)
            & panel[companion].notna().to_numpy()
        )

        for horizon in (1, 4):
            future = panel[outcome].groupby(
                analysis_stays, sort=False
            ).shift(-horizon).to_numpy(dtype=float)
            mask = (
                complete_running
                & (analysis_hours >= 12)
                & (analysis_hours <= 24 - horizon)
                & np.isfinite(future)
            )
            idx = np.flatnonzero(mask)
            XX = context.iloc[idx]
            dose = scaled_rate[idx]
            # Residualize both dose bases independently using the same folds.
            targets = np.column_stack([
                future[idx] - now[idx], dose, dose ** 2
            ])
            nuisance_predictions = np.empty_like(targets)
            folds_here = np.asarray(analysis_fold)[idx]

            for fold in fold_values:
                test = folds_here == fold
                for column in range(3):
                    nuisance = HistGradientBoostingRegressor(**params)
                    nuisance.fit(
                        XX.iloc[np.flatnonzero(~test)],
                        targets[~test, column]
                    )
                    nuisance_predictions[test, column] = nuisance.predict(
                        XX.iloc[np.flatnonzero(test)]
                    )

            residual = targets - nuisance_predictions
            y = residual[:, 0] - residual[:, 0].mean()
            R = residual[:, 1:] - residual[:, 1:].mean(axis=0)
            gram = R.T @ R
            inv_gram = np.linalg.pinv(gram)
            beta = inv_gram @ (R.T @ y)
            errors = y - R @ beta
            cluster = pd.DataFrame(
                R * errors[:, None], columns=["linear", "quadratic"]
            ).groupby(
                np.asarray(analysis_stays)[idx], sort=False
            ).sum().to_numpy()
            nclusters = len(cluster)
            covariance = (
                inv_gram @ (cluster.T @ cluster) @ inv_gram
                * nclusters / (nclusters - 1)
            )
            se = np.sqrt(np.maximum(np.diag(covariance), 0))
            wald = float(beta @ np.linalg.pinv(covariance) @ beta)

            result = {
                "drug": drug, "horizon": horizon, "n": len(idx),
                "rate_scale": scale,
                "linear_beta": float(beta[0]),
                "linear_se": float(se[0]),
                "quadratic_beta": float(beta[1]),
                "quadratic_se": float(se[1]),
                "joint_p": float(chi2.sf(wald, 2)),
                "basis_condition_number": float(np.linalg.cond(gram)),
            }
            # Local slopes only: do not extrapolate doubled doses beyond support.
            for label, quantile in [("q25", 0.25), ("q75", 0.75)]:
                dose_q = float(np.quantile(dose, quantile))
                derivative = np.array([1.0, 2.0 * dose_q]) / scale
                result[label + "_rate"] = dose_q * scale
                result[label + "_slope"] = float(derivative @ beta)
                result[label + "_slope_se"] = float(
                    np.sqrt(max(derivative @ covariance @ derivative, 0))
                )
            quadratic_checks.append(result)

print("=== QUADRATIC RESPONSE CHECKS ===")
print(pd.DataFrame(quadratic_checks).round(5).to_string(index=False))
print("Elapsed seconds:", round(time.time() - started, 1))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 14: nonlinear magnitude checks\n"
        "Cross-fitted linear/quadratic dose-response moments, restricted "
        "to late-period continuing positive infusions with both current "
        "sensors observed. Outcome is future minus current reading; "
        "controls are pre-action history and the other current drug. "
        "These are single-action association checks, not estimates of "
        "the evaluation's sustained-dose rollout. No extrapolation of "
        "quadratic curves to doubled doses is used.\n"
        + json.dumps(quadratic_checks) + "\n"
    )
```