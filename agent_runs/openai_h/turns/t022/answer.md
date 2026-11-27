Across all 400 stays, factual NLL increased by 0.000307 nats—far below the limit. The delayed estimates show no consistent response in either target sensor. Before accepting zero effects, I’ll specifically test dose variation among existing infusions, including curvature: a clear response there would challenge the edit because evaluation changes rates, not treatment presence.

```python
import time
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

def audit_cluster_regression(features, outcome, stay_ids):
    design = np.column_stack([np.ones(len(outcome)), features])
    coef = np.linalg.lstsq(design, outcome, rcond=None)[0]
    error = outcome - design @ coef
    unique_ids, inverse_ids = np.unique(stay_ids, return_inverse=True)
    cluster_scores = np.zeros((len(unique_ids), design.shape[1]))
    np.add.at(cluster_scores, inverse_ids, design * error[:, None])
    bread = np.linalg.pinv(design.T @ design)
    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread
    covariance *= len(unique_ids) / (len(unique_ids) - 1)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    slopes = coef[1:]
    slope_covariance = covariance[1:, 1:]
    statistic = float(slopes @ np.linalg.pinv(slope_covariance) @ slopes)
    return {
        "n": len(outcome),
        "stays": len(unique_ids),
        "coefficients": slopes.tolist(),
        "cluster_se": standard_error[1:].tolist(),
        "joint_null_p": float(chi2.sf(statistic, len(slopes))),
    }

audit_positive_results = {}
audit_positive_models = {}
audit_positive_params = dict(strong_params)
audit_positive_params.update(
    max_iter=250,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=10.0,
    early_stopping=False,
    random_state=81081,
)
audit_start_time = time.time()

with threadpool_limits(limits=8):
    for audit_j, audit_drug, audit_event, audit_severity_col in [
        (0, "insulin", "glucose_finger", "glucose_finger_last"),
        (1, "norepinephrine", "map_arterial", "map_arterial_last"),
    ]:
        audit_next_y = panel.groupby(level="stay_id")[audit_event].shift(-1).to_numpy()
        audit_dose = D[:, audit_j].astype(np.float64)
        audit_valid = (
            np.asarray(eligible, dtype=bool)
            & (audit_dose > 0)
            & np.isfinite(audit_next_y)
        )
        audit_fit_mask = audit_valid & ~np.asarray(held, dtype=bool)
        audit_test_mask = audit_valid & np.asarray(held, dtype=bool)
        audit_x_fit = Xrich[audit_fit_mask]
        audit_x_test = Xrich[audit_test_mask]
        audit_test_ids = stay_array[audit_test_mask]

        audit_predictions = []
        for audit_target_label, audit_target in [
            ("dose", audit_dose),
            ("dose_squared", audit_dose ** 2),
            ("next_reading", audit_next_y),
        ]:
            audit_model = HistGradientBoostingRegressor(**audit_positive_params)
            audit_model.fit(audit_x_fit, audit_target[audit_fit_mask])
            audit_predictions.append(audit_model.predict(audit_x_test))
            audit_positive_models[(audit_drug, audit_target_label)] = audit_model

        audit_residual_basis = np.column_stack([
            audit_dose[audit_test_mask] - audit_predictions[0],
            audit_dose[audit_test_mask] ** 2 - audit_predictions[1],
        ])
        audit_residual_y = audit_next_y[audit_test_mask] - audit_predictions[2]
        audit_linear = audit_cluster_regression(
            audit_residual_basis[:, :1], audit_residual_y, audit_test_ids
        )
        audit_quadratic = audit_cluster_regression(
            audit_residual_basis, audit_residual_y, audit_test_ids
        )

        audit_severity = Hrich[audit_severity_col].to_numpy()
        audit_cuts = np.nanquantile(audit_severity[audit_fit_mask], [.25, .5, .75])
        audit_test_severity = audit_severity[audit_test_mask]
        audit_strata = np.searchsorted(audit_cuts, audit_test_severity)
        audit_stratum_results = {}
        for audit_q in range(4):
            audit_sub = (audit_strata == audit_q) & np.isfinite(audit_test_severity)
            if audit_sub.sum() >= 100:
                audit_stratum_results[str(audit_q + 1)] = audit_cluster_regression(
                    audit_residual_basis[audit_sub, :1],
                    audit_residual_y[audit_sub],
                    audit_test_ids[audit_sub],
                )

        audit_positive_results[audit_drug] = {
            "outcome": audit_event,
            "training_rows": int(audit_fit_mask.sum()),
            "held_out_positive_dose_rows": int(audit_test_mask.sum()),
            "dose_residual_sd": float(np.std(audit_residual_basis[:, 0])),
            "linear": audit_linear,
            "linear_and_quadratic": audit_quadratic,
            "pretreatment_severity_quartiles": audit_stratum_results,
        }
        print(audit_drug)
        print(json.dumps(audit_positive_results[audit_drug], indent=2))
        print("Elapsed seconds:", round(time.time() - audit_start_time, 1))

with Path("/work/notes.md").open("a") as audit_f:
    audit_f.write(
        "\n## Python run 14 / forward passes 7: intensive-margin challenge\n"
        "- All 400 tuning stays: original NLL 1.48944949, edited 1.48975656; "
        "increase 0.00030707, stay-cluster approximate 95% interval "
        "[-0.00000357, 0.00061770]. After hour 12 the increase is 0.000145.\n"
        "- Existing target-sensor local projections at horizons 2/3/6/12 "
        "show no consistently signed, precisely estimated dose response.\n"
        "- New diagnostic uses only positive infusions, disjoint training/held-out "
        "stays, pretreatment histories, linear/quadratic dose residualization, "
        "and stay-cluster uncertainty. No model weights were changed.\n"
        "- Intensive-margin results: " + json.dumps(audit_positive_results) + "\n"
    )
```