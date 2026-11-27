The three-hour association survives on independent stays: about +0.14 mg/dL per insulin rate unit, while the immediate estimate remains near zero. That weakens the exact-null hypothesis. I’ll jointly test current and past doses against next-hour glucose, controlling measured history. A stable lagged response would justify a nonzero edit; disappearance under this adjustment would favor retaining dose invariance.

```python
import time
import numpy as np
import pandas as pd
import json
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

t0 = time.time()
insulin_lag_predictions = np.full(insulin_lag_basis.shape, np.nan, dtype=np.float64)
insulin_lag_y_prediction = np.full(len(Hrich), np.nan)
lag_params = dict(audit_positive_params)

with threadpool_limits(limits=8):
    for fold in range(5):
        fit_mask = insulin_lag_valid & (insulin_rep_fold != fold)
        test_mask = insulin_lag_valid & (insulin_rep_fold == fold)
        assert fit_mask.sum() > 10000 and test_mask.sum() > 1000
        for j in range(insulin_lag_basis.shape[1]):
            estimator = HistGradientBoostingRegressor(**lag_params)
            estimator.fit(insulin_lag_X[fit_mask], insulin_lag_basis[fit_mask, j])
            insulin_lag_predictions[test_mask, j] = estimator.predict(insulin_lag_X[test_mask])
        estimator = HistGradientBoostingRegressor(**lag_params)
        estimator.fit(insulin_lag_X[fit_mask], insulin_lag_target[fit_mask])
        insulin_lag_y_prediction[test_mask] = estimator.predict(insulin_lag_X[test_mask])
        print("Completed fold", fold, "elapsed_seconds", round(time.time() - t0, 1))

insulin_lag_residual_d = insulin_lag_basis - insulin_lag_predictions
insulin_lag_residual_y = insulin_lag_target - insulin_lag_y_prediction

linear_control_names = [
    "intercept", "hour",
    "glucose_finger_raw_current", "glucose_lab_raw_current",
    "glucose_lab_observed",
    *[f"G_lag{j}" for j in range(1, 7)],
    "map_arterial_raw_current", "map_cuff_raw_current",
]
insulin_lag_linear_controls = np.nan_to_num(
    Hrich[linear_control_names].to_numpy(dtype=np.float64),
    nan=0.0, posinf=0.0, neginf=0.0,
)

# These contrasts describe a one-step conditional effect with readings held
# fixed, NOT a simulated counterfactual rollout.
changed_lags = (insulin_rep_hour[:, None] - np.arange(7)[None, :] >= 12)
insulin_lag_x2_rows = np.column_stack([
    insulin_lag_rates * changed_lags,
    3.0 * insulin_lag_rates ** 2 * changed_lags,
])
insulin_lag_half_rows = np.column_stack([
    -0.5 * insulin_lag_rates * changed_lags,
    -0.75 * insulin_lag_rates ** 2 * changed_lags,
])
insulin_after12_seen = (
    pd.Series(np.where(insulin_rep_hour >= 12, insulin_rep_d, 0.0))
    .groupby(insulin_rep_ids, sort=False).cumsum().to_numpy() > 0
)

def clustered_lag_fit(mask, quadratic=True, add_linear_controls=False):
    n_dose = 14 if quadratic else 7
    design = insulin_lag_residual_d[mask, :n_dose]
    if add_linear_controls:
        design = np.column_stack([design, insulin_lag_linear_controls[mask]])
    target = insulin_lag_residual_y[mask]
    ids = insulin_rep_ids[mask]
    assert np.isfinite(design).all() and np.isfinite(target).all()

    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ beta
    unique_ids, groups = np.unique(ids, return_inverse=True)
    scores = np.zeros((len(unique_ids), design.shape[1]))
    np.add.at(scores, groups, design * residual[:, None])
    bread = np.linalg.pinv(design.T @ design, rcond=1e-12)
    correction = (
        len(unique_ids) / (len(unique_ids) - 1)
        * (len(target) - 1) / (len(target) - design.shape[1])
    )
    covariance = correction * bread @ (scores.T @ scores) @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    dose_covariance = covariance[:n_dose, :n_dose]
    wald = beta[:n_dose] @ np.linalg.pinv(dose_covariance) @ beta[:n_dose]
    rank = np.linalg.matrix_rank(dose_covariance)

    evaluation_mask = (
        mask & (insulin_rep_hour >= 12) & insulin_after12_seen
    )
    contrast_results = {}
    for label, row_contrast in (
        ("x2_conditional_shift", insulin_lag_x2_rows),
        ("half_conditional_shift", insulin_lag_half_rows),
    ):
        per_stay = pd.DataFrame(
            row_contrast[evaluation_mask, :n_dose]
        ).groupby(insulin_rep_ids[evaluation_mask], sort=False).mean()
        contrast = np.zeros(len(beta))
        contrast[:n_dose] = per_stay.mean(axis=0).to_numpy()
        contrast_results[label] = {
            "estimate_mg_dL": float(contrast @ beta),
            "cluster_se": float(np.sqrt(max(contrast @ covariance @ contrast, 0.0))),
        }

    summary = {
        "n": int(mask.sum()),
        "stays": int(len(unique_ids)),
        "joint_dose_p": float(chi2.sf(wald, rank)),
        "current_linear_coefficient": float(beta[0]),
        "current_linear_se": float(standard_error[0]),
        "lag_linear_coefficients": beta[:7].tolist(),
        "lag_linear_se": standard_error[:7].tolist(),
        **contrast_results,
    }
    if quadratic:
        summary["lag_quadratic_coefficients"] = beta[7:14].tolist()
        summary["lag_quadratic_se"] = standard_error[7:14].tolist()
    return summary

insulin_lag_results = {}
with threadpool_limits(limits=8):
    for label, mask in {
        "all_stays": insulin_lag_valid,
        "independent_stays": insulin_lag_valid & (insulin_rep_fold != 0),
    }.items():
        insulin_lag_results[label] = {}
        for specification, quadratic, controls in (
            ("linear", False, False),
            ("quadratic", True, False),
            ("quadratic_with_smooth_history_controls", True, True),
        ):
            result = clustered_lag_fit(mask, quadratic, controls)
            insulin_lag_results[label][specification] = result
            compact = {
                k: v for k, v in result.items()
                if k not in {
                    "lag_linear_coefficients", "lag_linear_se",
                    "lag_quadratic_coefficients", "lag_quadratic_se",
                }
            }
            print(label, specification, json.dumps(compact))

print("\n=== FULL ADJUSTED LAG PROFILE ===")
print(json.dumps(
    insulin_lag_results["all_stays"]["quadratic_with_smooth_history_controls"],
    indent=2))
print("Elapsed seconds:", round(time.time() - t0, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 23 / forward passes 7: joint insulin lag-dose diagnostic\n"
            "- Post-hour-12 three-hour association replicated on independent stays (+0.1408, SE 0.0687 mg/dL per rate unit); this is counter-evidence to exact zero.\n"
            "- Cross-fitted next-hour glucose-change nuisance models jointly residualize current and six past dose magnitudes, including quadratic terms.\n"
            "- Dose-presence history is retained; own lagged dose magnitudes are removed from nuisance inputs.\n"
            "- Compared additional smooth measured-history controls to reduce residual tree-binning bias.\n"
            "- Conditional-shift contrasts hold readings fixed and are NOT rollout estimates.\n"
            "- Results: " + json.dumps(insulin_lag_results) + "\n"
            "- No weight changes.\n")
```